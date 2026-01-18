import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
class ReferenceProperty(Property):
    data_type = Key
    type_name = 'Reference'

    def __init__(self, reference_class=None, collection_name=None, verbose_name=None, name=None, default=None, required=False, validator=None, choices=None, unique=False):
        super(ReferenceProperty, self).__init__(verbose_name, name, default, required, validator, choices, unique)
        self.reference_class = reference_class
        self.collection_name = collection_name

    def __get__(self, obj, objtype):
        if obj:
            value = getattr(obj, self.slot_name)
            if value == self.default_value():
                return value
            if isinstance(value, six.string_types):
                value = self.reference_class(value)
                setattr(obj, self.name, value)
            return value

    def __set__(self, obj, value):
        """Don't allow this object to be associated to itself
        This causes bad things to happen"""
        if value is not None and (obj.id == value or (hasattr(value, 'id') and obj.id == value.id)):
            raise ValueError('Can not associate an object with itself!')
        return super(ReferenceProperty, self).__set__(obj, value)

    def __property_config__(self, model_class, property_name):
        super(ReferenceProperty, self).__property_config__(model_class, property_name)
        if self.collection_name is None:
            self.collection_name = '%s_%s_set' % (model_class.__name__.lower(), self.name)
        if hasattr(self.reference_class, self.collection_name):
            raise ValueError('duplicate property: %s' % self.collection_name)
        setattr(self.reference_class, self.collection_name, _ReverseReferenceProperty(model_class, property_name, self.collection_name))

    def check_uuid(self, value):
        t = value.split('-')
        if len(t) != 5:
            raise ValueError

    def check_instance(self, value):
        try:
            obj_lineage = value.get_lineage()
            cls_lineage = self.reference_class.get_lineage()
            if obj_lineage.startswith(cls_lineage):
                return
            raise TypeError('%s not instance of %s' % (obj_lineage, cls_lineage))
        except:
            raise ValueError('%s is not a Model' % value)

    def validate(self, value):
        if self.validator:
            self.validator(value)
        if self.required and value is None:
            raise ValueError('%s is a required property' % self.name)
        if value == self.default_value():
            return
        if not isinstance(value, six.string_types):
            self.check_instance(value)