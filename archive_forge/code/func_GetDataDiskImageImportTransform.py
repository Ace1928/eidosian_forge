from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
def GetDataDiskImageImportTransform(value):
    """Returns empty DataDiskImageImport entry."""
    del value
    data_disk_image_import = _GetMessageClass('DataDiskImageImport')
    return data_disk_image_import()