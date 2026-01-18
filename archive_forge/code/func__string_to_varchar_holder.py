import json
import os
import pyarrow as pa
import pyarrow.jvm as pa_jvm
import pytest
import sys
import xml.etree.ElementTree as ET
def _string_to_varchar_holder(ra, string):
    nvch_cls = 'org.apache.arrow.vector.holders.NullableVarCharHolder'
    holder = jpype.JClass(nvch_cls)()
    if string is None:
        holder.isSet = 0
    else:
        holder.isSet = 1
        value = jpype.JClass('java.lang.String')('string')
        std_charsets = jpype.JClass('java.nio.charset.StandardCharsets')
        bytes_ = value.getBytes(std_charsets.UTF_8)
        holder.buffer = ra.buffer(len(bytes_))
        holder.buffer.setBytes(0, bytes_, 0, len(bytes_))
        holder.start = 0
        holder.end = len(bytes_)
    return holder