from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import xml
from xml.dom.minidom import parseString
from xml.sax import _exceptions as SaxExceptions
import six
import boto
from boto import handler
from boto.s3.tagging import Tags
from gslib.exception import CommandException
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.retry_util import Retry
from gslib.utils.constants import UTF8
def _LabelDictFromXmlString(self, xml_str):
    label_dict = {}
    tags_list = Tags()
    h = handler.XmlHandler(tags_list, None)
    try:
        xml.sax.parseString(xml_str, h)
    except SaxExceptions.SAXParseException as e:
        raise CommandException('Requested labels/tagging config is invalid: %s at line %s, column %s' % (e.getMessage(), e.getLineNumber(), e.getColumnNumber()))
    for tagset_list in tags_list:
        for tag in tagset_list:
            label_dict[tag.key] = tag.value
    return label_dict