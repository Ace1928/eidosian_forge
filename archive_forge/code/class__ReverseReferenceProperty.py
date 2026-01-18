import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
class _ReverseReferenceProperty(Property):
    data_type = Query
    type_name = 'query'

    def __init__(self, model, prop, name):
        self.__model = model
        self.__property = prop
        self.collection_name = prop
        self.name = name
        self.item_type = model

    def __get__(self, model_instance, model_class):
        """Fetches collection of model instances of this collection property."""
        if model_instance is not None:
            query = Query(self.__model)
            if isinstance(self.__property, list):
                props = []
                for prop in self.__property:
                    props.append('%s =' % prop)
                return query.filter(props, model_instance)
            else:
                return query.filter(self.__property + ' =', model_instance)
        else:
            return self

    def __set__(self, model_instance, value):
        """Not possible to set a new collection."""
        raise ValueError('Virtual property is read-only')