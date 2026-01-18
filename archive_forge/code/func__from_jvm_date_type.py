import pyarrow as pa
def _from_jvm_date_type(jvm_type):
    """
    Convert a JVM date type to its Python equivalent

    Parameters
    ----------
    jvm_type: org.apache.arrow.vector.types.pojo.ArrowType$Date

    Returns
    -------
    typ: pyarrow.DataType
    """
    day_unit = jvm_type.getUnit().toString()
    if day_unit == 'DAY':
        return pa.date32()
    elif day_unit == 'MILLISECOND':
        return pa.date64()