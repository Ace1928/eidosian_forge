from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import operator
from googlecloudsdk.api_lib.kuberun import service
from googlecloudsdk.api_lib.kuberun import traffic
import six
class TrafficTargetPair(object):
    """Holder for TrafficTarget status information.

  The representation of the status of traffic for a service
  includes:
    o User requested assignments (spec.traffic)
    o Actual assignments (status.traffic)

  Each of spec.traffic and status.traffic may contain multiple traffic targets
  that reference the same revision, either directly by name or indirectly by
  referencing the latest ready revision.

  The spec and status traffic targets for a revision may differ after a failed
  traffic update or during a successful one. A TrafficTargetPair holds all
  spec and status TrafficTargets that reference the same revision by name or
  reference the latest ready revision. Both the spec and status traffic targets
  can be empty.

  The latest revision can be included in the spec traffic targets
  two ways
    o by revisionName
    o by setting latestRevision to True.

  Attributes:
    key: Either the referenced revision name or 'LATEST' if the traffic targets
      reference the latest ready revision.
    latestRevision: Boolean indicating if the traffic targets reference the
      latest ready revision.
    revisionName: The name of the revision referenced by these traffic targets.
    specPercent: The percent of traffic allocated to the referenced revision in
      the service's spec.
    statusPercent: The percent of traffic allocated to the referenced revision
      in the service's status.
    specTags: Tags assigned to the referenced revision in the service's spec as
      a comma and space separated string.
    statusTags: Tags assigned to the referenced revision in the service's status
      as a comma and space separated string.
    urls: A list of urls that directly address the referenced revision.
    tags: A list of TrafficTag objects containing both the spec and status state
      for each traffic tag.
    displayPercent: Human-readable representation of the current percent
      assigned to the referenced revision.
    displayRevisionId: Human-readable representation of the name of the
      referenced revision.
    displayTags: Human-readable representation of the current tags assigned to
      the referenced revision.
    serviceUrl: The main URL for the service.
  """

    def __init__(self, spec_targets, status_targets, revision_name, latest, service_url=''):
        """Creates a new TrafficTargetPair.

    Args:
      spec_targets: A list of spec TrafficTargets that all reference the same
        revision, either by name or the latest ready.
      status_targets: A list of status TrafficTargets that all reference the
        same revision, either by name or the latest ready.
      revision_name: The name of the revision referenced by the traffic targets.
      latest: A boolean indicating if these traffic targets reference the latest
        ready revision.
      service_url: The main URL for the service. Optional.

    Returns:
      A new TrafficTargetPair instance.
    """
        self._spec_targets = spec_targets
        self._status_targets = status_targets
        self._revision_name = revision_name
        self._latest = latest
        self._service_url = service_url
        self._tags = None

    @property
    def latestRevision(self):
        """Returns true if the traffic targets reference the latest revision."""
        return self._latest

    @property
    def revisionName(self):
        return self._revision_name

    @property
    def specPercent(self):
        if self._spec_targets:
            return six.text_type(_SumPercent(self._spec_targets))
        else:
            return _MISSING_PERCENT_OR_TAGS

    @property
    def statusPercent(self):
        if self._status_targets:
            return six.text_type(_SumPercent(self._status_targets))
        else:
            return _MISSING_PERCENT_OR_TAGS

    @property
    def specTags(self):
        spec_tags = _TAGS_JOIN_STRING.join(sorted((t.tag for t in self._spec_targets if t.tag)))
        return spec_tags if spec_tags else _MISSING_PERCENT_OR_TAGS

    @property
    def statusTags(self):
        status_tags = _TAGS_JOIN_STRING.join(sorted((t.tag for t in self._status_targets if t.tag)))
        return status_tags if status_tags else _MISSING_PERCENT_OR_TAGS

    @property
    def urls(self):
        return sorted((t.url for t in self._status_targets if t.url))

    @property
    def tags(self):
        if self._tags is None:
            self._ExtractTags()
        return self._tags

    def _ExtractTags(self):
        """Extracts the traffic tag state from spec and status into TrafficTags."""
        tags = {}
        for spec_target in self._spec_targets:
            if not spec_target.tag:
                continue
            tags[spec_target.tag] = TrafficTag(spec_target.tag, in_spec=True)
        for status_target in self._status_targets:
            if not status_target.tag:
                continue
            if status_target.tag in tags:
                tag = tags[status_target.tag]
            else:
                tag = tags.setdefault(status_target.tag, TrafficTag(status_target.tag))
            tag.url = status_target.url if status_target.url is not None else ''
            tag.inStatus = True
        self._tags = sorted(tags.values(), key=operator.attrgetter('tag'))

    @property
    def displayPercent(self):
        """Returns human readable revision percent."""
        if self.statusPercent == self.specPercent:
            return _FormatPercentage(self.statusPercent)
        else:
            return '{:4} (currently {})'.format(_FormatPercentage(self.specPercent), _FormatPercentage(self.statusPercent))

    @property
    def displayRevisionId(self):
        """Returns human readable revision identifier."""
        if self.latestRevision:
            return '%s (currently %s)' % (traffic.GetKey(self), self.revisionName)
        else:
            return self.revisionName

    @property
    def displayTags(self):
        spec_tags = self.specTags
        status_tags = self.statusTags
        if spec_tags == status_tags:
            return status_tags if status_tags != _MISSING_PERCENT_OR_TAGS else ''
        else:
            return '{} (currently {})'.format(spec_tags, status_tags)

    @property
    def serviceUrl(self):
        """The main URL for the service."""
        return self._service_url