import io
import torch
from torch.package import Importer, OrderedImporter, PackageImporter, sys_importer
from torch.package._package_pickler import create_pickler
from torch.package._package_unpickler import PackageUnpickler
from torch.serialization import _maybe_decode_ascii
def _save_storages(importer, obj):
    serialized_storages = []
    serialized_dtypes = []
    importer = importer if isinstance(importer, torch.package.PackageImporter) else None
    importers: Importer
    if importer is not None:
        importers = OrderedImporter(importer, sys_importer)
    else:
        importers = sys_importer

    def persistent_id(obj):
        if torch.is_storage(obj) or isinstance(obj, torch.storage.TypedStorage):
            if isinstance(obj, torch.storage.TypedStorage):
                storage = obj._untyped_storage
                dtype = obj.dtype
            else:
                storage = obj
                dtype = torch.uint8
            serialized_storages.append(obj)
            serialized_dtypes.append(dtype)
            return ('storage', len(serialized_storages) - 1)
        if hasattr(obj, '__reduce_deploy__'):
            if _serialized_reduces.get(id(obj)) is None:
                _serialized_reduces[id(obj)] = ('reduce_deploy', id(obj), *obj.__reduce_deploy__(importers))
            return _serialized_reduces[id(obj)]
        return None
    data_buf = io.BytesIO()
    pickler = create_pickler(data_buf, importers)
    pickler.persistent_id = persistent_id
    pickler.dump(obj)
    data_value = data_buf.getvalue()
    return (data_value, serialized_storages, serialized_dtypes, importer.zip_reader if importer else None)