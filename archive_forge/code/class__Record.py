from struct import pack, unpack, calcsize, error, Struct
import os
import sys
import time
import array
import tempfile
import logging
import io
from datetime import date
import zipfile
class _Record(list):
    """
    A class to hold a record. Subclasses list to ensure compatibility with
    former work and to reuse all the optimizations of the builtin list.
    In addition to the list interface, the values of the record
    can also be retrieved using the field's name. For example if the dbf contains
    a field ID at position 0, the ID can be retrieved with the position, the field name
    as a key, or the field name as an attribute.

    >>> # Create a Record with one field, normally the record is created by the Reader class
    >>> r = _Record({'ID': 0}, [0])
    >>> print(r[0])
    >>> print(r['ID'])
    >>> print(r.ID)
    """

    def __init__(self, field_positions, values, oid=None):
        """
        A Record should be created by the Reader class

        :param field_positions: A dict mapping field names to field positions
        :param values: A sequence of values
        :param oid: The object id, an int (optional)
        """
        self.__field_positions = field_positions
        if oid is not None:
            self.__oid = oid
        else:
            self.__oid = -1
        list.__init__(self, values)

    def __getattr__(self, item):
        """
        __getattr__ is called if an attribute is used that does
        not exist in the normal sense. For example r=Record(...), r.ID
        calls r.__getattr__('ID'), but r.index(5) calls list.index(r, 5)
        :param item: The field name, used as attribute
        :return: Value of the field
        :raises: AttributeError, if item is not a field of the shapefile
                and IndexError, if the field exists but the field's 
                corresponding value in the Record does not exist
        """
        try:
            index = self.__field_positions[item]
            return list.__getitem__(self, index)
        except KeyError:
            raise AttributeError('{} is not a field name'.format(item))
        except IndexError:
            raise IndexError('{} found as a field but not enough values available.'.format(item))

    def __setattr__(self, key, value):
        """
        Sets a value of a field attribute
        :param key: The field name
        :param value: the value of that field
        :return: None
        :raises: AttributeError, if key is not a field of the shapefile
        """
        if key.startswith('_'):
            return list.__setattr__(self, key, value)
        try:
            index = self.__field_positions[key]
            return list.__setitem__(self, index, value)
        except KeyError:
            raise AttributeError('{} is not a field name'.format(key))

    def __getitem__(self, item):
        """
        Extends the normal list item access with
        access using a fieldname

        For example r['ID'], r[0]
        :param item: Either the position of the value or the name of a field
        :return: the value of the field
        """
        try:
            return list.__getitem__(self, item)
        except TypeError:
            try:
                index = self.__field_positions[item]
            except KeyError:
                index = None
        if index is not None:
            return list.__getitem__(self, index)
        else:
            raise IndexError('"{}" is not a field name and not an int'.format(item))

    def __setitem__(self, key, value):
        """
        Extends the normal list item access with
        access using a fieldname

        For example r['ID']=2, r[0]=2
        :param key: Either the position of the value or the name of a field
        :param value: the new value of the field
        """
        try:
            return list.__setitem__(self, key, value)
        except TypeError:
            index = self.__field_positions.get(key)
            if index is not None:
                return list.__setitem__(self, index, value)
            else:
                raise IndexError('{} is not a field name and not an int'.format(key))

    @property
    def oid(self):
        """The index position of the record in the original shapefile"""
        return self.__oid

    def as_dict(self, date_strings=False):
        """
        Returns this Record as a dictionary using the field names as keys
        :return: dict
        """
        dct = dict(((f, self[i]) for f, i in self.__field_positions.items()))
        if date_strings:
            for k, v in dct.items():
                if isinstance(v, date):
                    dct[k] = '{:04d}{:02d}{:02d}'.format(v.year, v.month, v.day)
        return dct

    def __repr__(self):
        return 'Record #{}: {}'.format(self.__oid, list(self))

    def __dir__(self):
        """
        Helps to show the field names in an interactive environment like IPython.
        See: http://ipython.readthedocs.io/en/stable/config/integrating.html

        :return: List of method names and fields
        """
        default = list(dir(type(self)))
        fnames = list(self.__field_positions.keys())
        return default + fnames