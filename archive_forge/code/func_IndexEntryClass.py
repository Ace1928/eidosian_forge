import collections
import functools
import operator
from ovs.db import data
def IndexEntryClass(table):
    """Create a class used represent Rows in indexes

    ovs.db.idl.Row, being inherently tied to transaction processing and being
    initialized with dicts of Datums, is not really useable as an object to
    pass to and store in indexes. This method will create a class named after
    the table's name that is initialized with that Table Row's default values.
    For example:

    Port = IndexEntryClass(idl.tables['Port'])

    will create a Port class. This class can then be used to search custom
    indexes. For example:

    for port in idx.iranage(Port(name="test1"), Port(name="test9")):
       ...
    """

    def defaults_uuid_to_row(atom, base):
        return atom.value
    columns = ['uuid'] + list(table.columns.keys())
    cls = collections.namedtuple(table.name, columns)
    cls._table = table
    cls.__new__.__defaults__ = (None,) + tuple((data.Datum.default(c.type).to_python(defaults_uuid_to_row) for c in table.columns.values()))
    return cls