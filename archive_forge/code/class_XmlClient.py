from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage.s3_xml import client as s3_xml_client
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core import properties
class XmlClient(s3_xml_client.S3XmlClient):
    """GCS XML client."""
    scheme = storage_url.ProviderPrefix.GCS

    def __init__(self):
        storage = properties.VALUES.storage
        self.access_key_id = storage.gs_xml_access_key_id.Get()
        self.access_key_secret = storage.gs_xml_secret_access_key.Get()
        self.endpoint_url = storage.gs_xml_endpoint_url.Get()
        self.client = self.create_client()