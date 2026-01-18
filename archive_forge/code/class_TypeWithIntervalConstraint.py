from googlecloudsdk.command_lib.concepts import concept_managers
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import re
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.command_lib.concepts import base
from googlecloudsdk.command_lib.concepts import dependency_managers
from googlecloudsdk.command_lib.concepts import exceptions
from googlecloudsdk.command_lib.concepts import names
from googlecloudsdk.core.util import scaled_integer
from googlecloudsdk.core.util import semver
from googlecloudsdk.core.util import times
import six
class TypeWithIntervalConstraint(ConceptType):
    """Concept type with value interval constraints.

  Validates that a ConceptType value is within the interval defined by min and
  max endpoints. A missing min or max endpoint indicates that there is no min or
  max value, respectively.

  Attributes:
    _min_endpoint: Endpoint, the minimum value interval endpoint.
    _max_endpoint: Endpoint, the maximum value interval endpoint.
    _constraint_kind: string, the interval value type name.
    _convert_endpoint: f(str)=>x, converts an endpoint string to a value.
    _convert_interval: f(str)=>x, converts an interval value to a value.
    _display_endpoint: f(value)=>str, displays an interval endpoint.
  """

    def __init__(self, name, min_endpoint=None, max_endpoint=None, constraint_kind=None, convert_endpoint=None, convert_value=None, display_endpoint=None, **kwargs):
        super(TypeWithIntervalConstraint, self).__init__(name, **kwargs)
        self._min_endpoint = min_endpoint
        self._max_endpoint = max_endpoint
        self._kind = constraint_kind or 'value'
        self._convert_endpoint = convert_endpoint or self.Convert
        self._display_endpoint = display_endpoint or self.Display
        self._convert_value = convert_value or (lambda x: x)
        if self._min_endpoint:
            self._ConvertEndpoint(self._min_endpoint, 'min endpoint')
        if self._max_endpoint:
            self._ConvertEndpoint(self._max_endpoint, 'max endpoint')

    def _ConvertEndpoint(self, endpoint, kind):
        """Declaration time endpoint conversion check."""
        message = None
        try:
            endpoint.value = self._convert_endpoint(endpoint.string)
            return
        except exceptions.ParseError as e:
            message = six.text_type(e).split('. ', 1)[1].rstrip('.')
        except (AttributeError, ValueError) as e:
            message = _SubException(e)
        raise exceptions.ConstraintError(self.GetPresentationName(), kind, endpoint.string, message + '.')

    def Constraints(self):
        """Returns the type constraints message text if any."""
        boundaries = []
        if self._min_endpoint:
            endpoint = self._min_endpoint.value
            if self._min_endpoint.closed:
                boundary = 'greater than or equal to'
            else:
                boundary = 'greater than'
            boundaries.append('{} {}'.format(boundary, self._display_endpoint(endpoint)))
        if self._max_endpoint:
            endpoint = self._max_endpoint.value
            if self._max_endpoint.closed:
                boundary = 'less than or equal to'
            else:
                boundary = 'less than'
            boundaries.append('{} {}'.format(boundary, self._display_endpoint(endpoint)))
        if not boundaries:
            return ''
        return 'The {} must be {}.'.format(self._kind, ' and '.join(boundaries))

    def Validate(self, value):
        value = self._convert_value(value)
        invalid = None
        if self._min_endpoint:
            endpoint = self._min_endpoint.value
            if self._min_endpoint.closed:
                if value < endpoint:
                    invalid = 'greater than or equal to'
            elif value <= endpoint:
                invalid = 'greater than'
        if not invalid and self._max_endpoint:
            endpoint = self._max_endpoint.value
            if self._max_endpoint.closed:
                if value > endpoint:
                    invalid = 'less than or equal to'
            elif value >= endpoint:
                invalid = 'less than'
        if invalid:
            raise exceptions.ValidationError(self.GetPresentationName(), '{}{} [{}] must be {} [{}].'.format(self._kind[0].upper(), self._kind[1:], self._display_endpoint(value), invalid, self._display_endpoint(endpoint)))