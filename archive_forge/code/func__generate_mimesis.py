from __future__ import annotations
import random
from packaging.version import Version
from dask.utils import import_required
def _generate_mimesis(field, schema_description, records_per_partition, seed):
    """Generate data for a single partition of a dask bag

    See Also
    --------
    _make_mimesis
    """
    import mimesis
    from mimesis.schema import Field, Schema
    field = Field(seed=seed, **field)
    schema_kwargs, create_kwargs = ({}, {})
    if Version(mimesis.__version__) < Version('9.0.0'):
        create_kwargs['iterations'] = 1
    else:
        schema_kwargs['iterations'] = 1
    schema = Schema(schema=lambda: schema_description(field), **schema_kwargs)
    return [schema.create(**create_kwargs)[0] for i in range(records_per_partition)]