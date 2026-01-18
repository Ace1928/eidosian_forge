from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def InsertDicomStreamConfig(api_version):
    """Returns a function that inserts a DicomStreamingConfig of a Dicom Store.

  Args:
    api_version: the version of the API that is currently being used.

  Returns:
    A DicomStreamConfig object with provided BigQueryDestinations.
  """

    def VersionedInsertDicomStreamConfig(arg):
        if not arg:
            return None
        bq_destinations = arg.split(',')
        messages = apis.GetMessagesModule('healthcare', api_version)
        stream_configs = []
        if api_version == 'v1alpha2':
            for dest in bq_destinations:
                stream_configs.append(messages.GoogleCloudHealthcareV1alpha2DicomStreamConfig(bigqueryDestination=messages.GoogleCloudHealthcareV1alpha2DicomBigQueryDestination(tableUri=dest)))
        elif api_version == 'v1beta1':
            for dest in bq_destinations:
                stream_configs.append(messages.GoogleCloudHealthcareV1beta1DicomStreamConfig(bigqueryDestination=messages.GoogleCloudHealthcareV1beta1DicomBigQueryDestination(tableUri=dest)))
        else:
            for dest in bq_destinations:
                stream_configs.append(messages.GoogleCloudHealthcareV1DicomStreamConfig(bigqueryDestination=messages.GoogleCloudHealthcareV1DicomBigQueryDestination(tableUri=dest)))
        return stream_configs
    return VersionedInsertDicomStreamConfig