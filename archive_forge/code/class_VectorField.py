from typing import List
from redis import DataError
class VectorField(Field):
    """
    Allows vector similarity queries against the value in this attribute.
    See https://oss.redis.com/redisearch/Vectors/#vector_fields.
    """

    def __init__(self, name: str, algorithm: str, attributes: dict, **kwargs):
        """
        Create Vector Field. Notice that Vector cannot have sortable or no_index tag,
        although it's also a Field.

        ``name`` is the name of the field.

        ``algorithm`` can be "FLAT" or "HNSW".

        ``attributes`` each algorithm can have specific attributes. Some of them
        are mandatory and some of them are optional. See
        https://oss.redis.com/redisearch/master/Vectors/#specific_creation_attributes_per_algorithm
        for more information.
        """
        sort = kwargs.get('sortable', False)
        noindex = kwargs.get('no_index', False)
        if sort or noindex:
            raise DataError("Cannot set 'sortable' or 'no_index' in Vector fields.")
        if algorithm.upper() not in ['FLAT', 'HNSW']:
            raise DataError("Realtime vector indexing supporting 2 Indexing Methods:'FLAT' and 'HNSW'.")
        attr_li = []
        for key, value in attributes.items():
            attr_li.extend([key, value])
        Field.__init__(self, name, args=[Field.VECTOR, algorithm, len(attr_li), *attr_li], **kwargs)