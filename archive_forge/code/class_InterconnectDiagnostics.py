from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectDiagnostics(_messages.Message):
    """Diagnostics information about the Interconnect connection, which
  contains detailed and current technical information about Google's side of
  the connection.

  Enums:
    BundleAggregationTypeValueValuesEnum: The aggregation type of the bundle
      interface.
    BundleOperationalStatusValueValuesEnum: The operational status of the
      bundle interface.

  Fields:
    arpCaches: A list of InterconnectDiagnostics.ARPEntry objects, describing
      individual neighbors currently seen by the Google router in the ARP
      cache for the Interconnect. This will be empty when the Interconnect is
      not bundled.
    bundleAggregationType: The aggregation type of the bundle interface.
    bundleOperationalStatus: The operational status of the bundle interface.
    links: A list of InterconnectDiagnostics.LinkStatus objects, describing
      the status for each link on the Interconnect.
    macAddress: The MAC address of the Interconnect's bundle interface.
  """

    class BundleAggregationTypeValueValuesEnum(_messages.Enum):
        """The aggregation type of the bundle interface.

    Values:
      BUNDLE_AGGREGATION_TYPE_LACP: LACP is enabled.
      BUNDLE_AGGREGATION_TYPE_STATIC: LACP is disabled.
    """
        BUNDLE_AGGREGATION_TYPE_LACP = 0
        BUNDLE_AGGREGATION_TYPE_STATIC = 1

    class BundleOperationalStatusValueValuesEnum(_messages.Enum):
        """The operational status of the bundle interface.

    Values:
      BUNDLE_OPERATIONAL_STATUS_DOWN: If bundleAggregationType is LACP: LACP
        is not established and/or all links in the bundle have DOWN
        operational status. If bundleAggregationType is STATIC: one or more
        links in the bundle has DOWN operational status.
      BUNDLE_OPERATIONAL_STATUS_UP: If bundleAggregationType is LACP: LACP is
        established and at least one link in the bundle has UP operational
        status. If bundleAggregationType is STATIC: all links in the bundle
        (typically just one) have UP operational status.
    """
        BUNDLE_OPERATIONAL_STATUS_DOWN = 0
        BUNDLE_OPERATIONAL_STATUS_UP = 1
    arpCaches = _messages.MessageField('InterconnectDiagnosticsARPEntry', 1, repeated=True)
    bundleAggregationType = _messages.EnumField('BundleAggregationTypeValueValuesEnum', 2)
    bundleOperationalStatus = _messages.EnumField('BundleOperationalStatusValueValuesEnum', 3)
    links = _messages.MessageField('InterconnectDiagnosticsLinkStatus', 4, repeated=True)
    macAddress = _messages.StringField(5)