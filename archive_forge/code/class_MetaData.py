import re
import datetime
import numpy as np
import csv
import ctypes
class MetaData:
    """Small container to keep useful information on a ARFF dataset.

    Knows about attributes names and types.

    Examples
    --------
    ::

        data, meta = loadarff('iris.arff')
        # This will print the attributes names of the iris.arff dataset
        for i in meta:
            print(i)
        # This works too
        meta.names()
        # Getting attribute type
        types = meta.types()

    Methods
    -------
    names
    types

    Notes
    -----
    Also maintains the list of attributes in order, i.e., doing for i in
    meta, where meta is an instance of MetaData, will return the
    different attribute names in the order they were defined.
    """

    def __init__(self, rel, attr):
        self.name = rel
        self._attributes = {a.name: a for a in attr}

    def __repr__(self):
        msg = ''
        msg += 'Dataset: %s\n' % self.name
        for i in self._attributes:
            msg += f"\t{i}'s type is {self._attributes[i].type_name}"
            if self._attributes[i].range:
                msg += ', range is %s' % str(self._attributes[i].range)
            msg += '\n'
        return msg

    def __iter__(self):
        return iter(self._attributes)

    def __getitem__(self, key):
        attr = self._attributes[key]
        return (attr.type_name, attr.range)

    def names(self):
        """Return the list of attribute names.

        Returns
        -------
        attrnames : list of str
            The attribute names.
        """
        return list(self._attributes)

    def types(self):
        """Return the list of attribute types.

        Returns
        -------
        attr_types : list of str
            The attribute types.
        """
        attr_types = [self._attributes[name].type_name for name in self._attributes]
        return attr_types