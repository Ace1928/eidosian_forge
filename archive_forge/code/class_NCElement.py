import io
import sys
import six
import types
from six import StringIO
from io import BytesIO
from lxml import etree
from ncclient import NCClientError
class NCElement(object):

    def __init__(self, result, transform_reply, huge_tree=False):
        self.__result = result
        self.__transform_reply = transform_reply
        self.__huge_tree = huge_tree
        if isinstance(transform_reply, types.FunctionType):
            self.__doc = self.__transform_reply(result._root)
        else:
            self.__doc = self.remove_namespaces(self.__result)

    def xpath(self, expression, namespaces={}):
        """Perform XPath navigation on an object

        Args:
            expression: A string representing a compliant XPath
                expression.
            namespaces: A dict of caller supplied prefix/xmlns to
                append to the static dict of XPath namespaces.
        Returns:
            A list of 'lxml.etree._Element' should a match on the
            expression be successful.  Otherwise, an empty list will
            be returned to the caller.
        """
        self.__expression = expression
        self.__namespaces = XPATH_NAMESPACES
        self.__namespaces.update(namespaces)
        return self.__doc.xpath(self.__expression, namespaces=self.__namespaces)

    def find(self, expression):
        """return result for a call to lxml ElementPath find()"""
        self.__expression = expression
        return self.__doc.find(self.__expression)

    def findtext(self, expression):
        """return result for a call to lxml ElementPath findtext()"""
        self.__expression = expression
        return self.__doc.findtext(self.__expression)

    def findall(self, expression):
        """return result for a call to lxml ElementPath findall()"""
        self.__expression = expression
        return self.__doc.findall(self.__expression)

    def __str__(self):
        """syntactic sugar for str() - alias to tostring"""
        if sys.version < '3':
            return self.tostring
        else:
            return self.tostring.decode('UTF-8')

    @property
    def tostring(self):
        """return a pretty-printed string output for rpc reply"""
        parser = etree.XMLParser(remove_blank_text=True, huge_tree=self.__huge_tree)
        outputtree = etree.XML(etree.tostring(self.__doc), parser)
        return etree.tostring(outputtree, pretty_print=True)

    @property
    def data_xml(self):
        """return an unmodified output for rpc reply"""
        return to_xml(self.__doc)

    def remove_namespaces(self, rpc_reply):
        """remove xmlns attributes from rpc reply"""
        self.__xslt = self.__transform_reply
        self.__parser = etree.XMLParser(remove_blank_text=True, huge_tree=self.__huge_tree)
        self.__xslt_doc = etree.parse(io.BytesIO(self.__xslt), self.__parser)
        self.__transform = etree.XSLT(self.__xslt_doc)
        self.__root = etree.fromstring(str(self.__transform(etree.parse(StringIO(str(rpc_reply)), parser=self.__parser))), parser=self.__parser)
        return self.__root