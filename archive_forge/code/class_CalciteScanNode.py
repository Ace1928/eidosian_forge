import abc
from .dataframe.utils import ColNameCodec
from .db_worker import DbTable
from .expr import BaseExpr
class CalciteScanNode(CalciteBaseNode):
    """
    A node to represent a scan operation.

    Scan operation can only be applied to physical tables.

    Parameters
    ----------
    modin_frame : HdkOnNativeDataframe
        A frame to scan. The frame should have a materialized table
        in HDK.

    Attributes
    ----------
    table : list of str
        A list holding a database name and a table name.
    fieldNames : list of str
        A list of columns to include into the scan.
    inputs : list
        An empty list existing for the sake of serialization
        simplicity. Has no meaning but is expected by HDK
        deserializer.
    """

    def __init__(self, modin_frame):
        assert modin_frame._partitions is not None
        table = modin_frame._partitions[0][0].get()
        assert isinstance(table, DbTable)
        super(CalciteScanNode, self).__init__('EnumerableTableScan')
        self.table = ['hdk', table.name]
        self.fieldNames = [ColNameCodec.encode(col) for col in modin_frame._table_cols] + ['rowid']
        self.inputs = []