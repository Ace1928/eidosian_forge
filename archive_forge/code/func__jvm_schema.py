import json
import os
import pyarrow as pa
import pyarrow.jvm as pa_jvm
import pytest
import sys
import xml.etree.ElementTree as ET
def _jvm_schema(jvm_spec, metadata=None):
    field = _jvm_field(jvm_spec)
    schema_cls = jpype.JClass('org.apache.arrow.vector.types.pojo.Schema')
    fields = jpype.JClass('java.util.ArrayList')()
    fields.add(field)
    if metadata:
        dct = jpype.JClass('java.util.HashMap')()
        for k, v in metadata.items():
            dct.put(k, v)
        return schema_cls(fields, dct)
    else:
        return schema_cls(fields)