import base64
import io
import logging
import smart_open.bytebuffer
import smart_open.constants
def _get_blob_client(client, container, blob):
    """
    Return an Azure BlobClient starting with any of BlobServiceClient,
    ContainerClient, or BlobClient plus container name and blob name.
    """
    if hasattr(client, 'get_container_client'):
        client = client.get_container_client(container)
    if hasattr(client, 'container_name') and client.container_name != container:
        raise ValueError("Client for %r doesn't match container %r" % (client.container_name, container))
    if hasattr(client, 'get_blob_client'):
        client = client.get_blob_client(blob)
    return client