import pyarrow as pa
def _from_jvm_time_type(jvm_type):
    """
    Convert a JVM time type to its Python equivalent.

    Parameters
    ----------
    jvm_type: org.apache.arrow.vector.types.pojo.ArrowType$Time

    Returns
    -------
    typ: pyarrow.DataType
    """
    time_unit = jvm_type.getUnit().toString()
    if time_unit == 'SECOND':
        assert jvm_type.getBitWidth() == 32
        return pa.time32('s')
    elif time_unit == 'MILLISECOND':
        assert jvm_type.getBitWidth() == 32
        return pa.time32('ms')
    elif time_unit == 'MICROSECOND':
        assert jvm_type.getBitWidth() == 64
        return pa.time64('us')
    elif time_unit == 'NANOSECOND':
        assert jvm_type.getBitWidth() == 64
        return pa.time64('ns')