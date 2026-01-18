from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import logging
import string
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
import six
class HttpErrorPayload(FormattableErrorPayload):
    """Converts apitools HttpError payload to an object.

  Attributes:
    api_name: The url api name.
    api_version: The url version.
    content: The dumped JSON content.
    details: A list of {'@type': TYPE, 'detail': STRING} typed details.
    domain_details: ErrorInfo metadata Indexed by domain.
    violations: map of subject to error message for that subject.
    field_violations: map of field name to error message for that field.
    error_info: content['error'].
    instance_name: The url instance name.
    message: The human readable error message.
    resource_item: The resource type.
    resource_name: The url resource name.
    resource_version: The resource version.
    status_code: The HTTP status code number.
    status_description: The status_code description.
    status_message: Context specific status message.
    type_details: ErrorDetails Indexed by type.
    url: The HTTP url.
    .<a>.<b>...: The <a>.<b>... attribute in the JSON content (synthesized in
      get_field()).

  Examples:
    error_format values and resulting output:

    'Error: [{status_code}] {status_message}{url?\\n{?}}{.debugInfo?\\n{?}}'

      Error: [404] Not found
      http://dotcom/foo/bar
      <content.debugInfo in yaml print format>

    'Error: {status_code} {details?\\n\\ndetails:\\n{?}}'

      Error: 404

      details:
      - foo
      - bar

     'Error [{status_code}] {status_message}\\n'
     '{.:value(details.detail.list(separator="\\n"))}'

       Error [400] Invalid request.
       foo
       bar
  """

    def __init__(self, http_error):
        super(HttpErrorPayload, self).__init__(http_error)
        self.api_name = ''
        self.api_version = ''
        self.details = []
        self.violations = {}
        self.field_violations = {}
        self.error_info = None
        self.instance_name = ''
        self.resource_item = ''
        self.resource_name = ''
        self.resource_version = ''
        self.url = ''
        if not isinstance(http_error, six.string_types):
            self._ExtractResponseAndJsonContent(http_error)
            self._ExtractUrlResourceAndInstanceNames(http_error)
            self.message = self._MakeGenericMessage()

    def _GetField(self, name):
        if name.startswith('field_violations.'):
            _, field = name.split('.', 1)
            value = self.field_violations.get(field)
        elif name.startswith('violations.'):
            _, subject = name.split('.', 1)
            value = self.violations.get(subject)
        else:
            name, value = super(HttpErrorPayload, self)._GetField(name)
        return (name, value)

    def _ExtractResponseAndJsonContent(self, http_error):
        """Extracts the response and JSON content from the HttpError."""
        response = getattr(http_error, 'response', None)
        if response:
            self.status_code = int(response.get('status', 0))
            self.status_description = encoding.Decode(response.get('reason', ''))
        content = encoding.Decode(http_error.content)
        try:
            self.content = _JsonSortedDict(json.loads(content))
            self.error_info = _JsonSortedDict(self.content['error'])
            if not self.status_code:
                self.status_code = int(self.error_info.get('code', 0))
            if not self.status_description:
                self.status_description = self.error_info.get('status', '')
            self.status_message = self.error_info.get('message', '')
            self.details = self.error_info.get('details', [])
            self.violations = self._ExtractViolations(self.details)
            self.field_violations = self._ExtractFieldViolations(self.details)
            self.type_details = self._IndexErrorDetailsByType(self.details)
            self.domain_details = self._IndexErrorInfoByDomain(self.details)
        except (KeyError, TypeError, ValueError):
            self.status_message = content
        except AttributeError:
            pass

    def _IndexErrorDetailsByType(self, details):
        """Extracts and indexes error details list by the type attribute."""
        type_map = collections.defaultdict(list)
        for item in details:
            error_type = item.get('@type', None)
            if error_type:
                error_type_suffix = error_type.split('.')[-1]
                type_map[error_type_suffix].append(item)
        return type_map

    def _IndexErrorInfoByDomain(self, details):
        """Extracts and indexes error info list by the domain attribute."""
        domain_map = collections.defaultdict(list)
        for item in details:
            error_type = item.get('@type', None)
            if error_type.endswith(ERROR_INFO_SUFFIX):
                domain = item.get('domain', None)
                if domain:
                    domain_map[domain].append(item)
        return domain_map

    def _ExtractUrlResourceAndInstanceNames(self, http_error):
        """Extracts the url resource type and instance names from the HttpError."""
        self.url = http_error.url
        if not self.url:
            return
        try:
            name, version, resource_path = resource_util.SplitEndpointUrl(self.url)
        except resource_util.InvalidEndpointException:
            return
        if name:
            self.api_name = name
        if version:
            self.api_version = version
        resource_parts = resource_path.split('/')
        if not 1 < len(resource_parts) < 4:
            return
        self.resource_name = resource_parts[0]
        instance_name = resource_parts[1]
        self.instance_name = instance_name.split('?')[0]
        self.resource_item = '{} instance'.format(self.resource_name)

    def _MakeDescription(self):
        """Makes description for error by checking which fields are filled in."""
        if self.status_code and self.resource_item and self.instance_name:
            if self.status_code == 403:
                return 'User [{0}] does not have permission to access {1} [{2}] (or it may not exist)'.format(properties.VALUES.core.account.Get(), self.resource_item, self.instance_name)
            if self.status_code == 404:
                return '{0} [{1}] not found'.format(self.resource_item.capitalize(), self.instance_name)
            if self.status_code == 409:
                if self.resource_name == 'projects':
                    return 'Resource in projects [{0}] is the subject of a conflict'.format(self.instance_name)
                else:
                    return '{0} [{1}] is the subject of a conflict'.format(self.resource_item.capitalize(), self.instance_name)
        return super(HttpErrorPayload, self)._MakeDescription()

    def _ExtractViolations(self, details):
        """Extracts a map of violations from the given error's details.

    Args:
      details: JSON-parsed details field from parsed json of error.

    Returns:
      Map[str, str] sub -> error description. The iterator of it is ordered by
      the order the subjects first appear in the errror.
    """
        results = collections.OrderedDict()
        for detail in details:
            if 'violations' not in detail:
                continue
            violations = detail['violations']
            if not isinstance(violations, list):
                continue
            sub = detail.get('subject')
            for violation in violations:
                try:
                    local_sub = violation.get('subject')
                    subject = sub or local_sub
                    if subject:
                        if subject in results:
                            results[subject] += '\n' + violation['description']
                        else:
                            results[subject] = violation['description']
                except (KeyError, TypeError):
                    pass
        return results

    def _ExtractFieldViolations(self, details):
        """Extracts a map of field violations from the given error's details.

    Args:
      details: JSON-parsed details field from parsed json of error.

    Returns:
      Map[str, str] field (in dotted format) -> error description.
      The iterator of it is ordered by the order the fields first
      appear in the error.
    """
        results = collections.OrderedDict()
        for deet in details:
            if 'fieldViolations' not in deet:
                continue
            violations = deet['fieldViolations']
            if not isinstance(violations, list):
                continue
            f = deet.get('field')
            for viol in violations:
                try:
                    local_f = viol.get('field')
                    field = f or local_f
                    if field:
                        if field in results:
                            results[field] += '\n' + viol['description']
                        else:
                            results[field] = viol['description']
                except (KeyError, TypeError):
                    pass
        return results