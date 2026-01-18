import pyarrow as pa
from pyarrow.util import _is_iterable, _stringify_path, _is_path_like
from pyarrow.compute import Expression, scalar, field  # noqa
def _ensure_partitioning(scheme):
    """
    Validate input and return a Partitioning(Factory).

    It passes None through if no partitioning scheme is defined.
    """
    if scheme is None:
        pass
    elif isinstance(scheme, str):
        scheme = partitioning(flavor=scheme)
    elif isinstance(scheme, list):
        scheme = partitioning(field_names=scheme)
    elif isinstance(scheme, (Partitioning, PartitioningFactory)):
        pass
    else:
        ValueError('Expected Partitioning or PartitioningFactory, got {}'.format(type(scheme)))
    return scheme