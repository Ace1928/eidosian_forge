import pyarrow as pa
from pyarrow.util import _is_iterable, _stringify_path, _is_path_like
from pyarrow.compute import Expression, scalar, field  # noqa
def _union_dataset(children, schema=None, **kwargs):
    if any((v is not None for v in kwargs.values())):
        raise ValueError('When passing a list of Datasets, you cannot pass any additional arguments')
    if schema is None:
        schema = pa.unify_schemas([child.schema for child in children])
    for child in children:
        if getattr(child, '_scan_options', None):
            raise ValueError('Creating an UnionDataset from filtered or projected Datasets is currently not supported. Union the unfiltered datasets and apply the filter to the resulting union.')
    children = [child.replace_schema(schema) for child in children]
    return UnionDataset(schema, children)