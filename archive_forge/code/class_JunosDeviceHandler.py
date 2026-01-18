import logging
import re
from lxml import etree
from lxml.etree import QName
from ncclient.operations.retrieve import GetSchemaReply
from .default import DefaultDeviceHandler
from ncclient.operations.third_party.juniper.rpc import GetConfiguration, LoadConfiguration, CompareConfiguration
from ncclient.operations.third_party.juniper.rpc import ExecuteRpc, Command, Reboot, Halt, Commit, Rollback
from ncclient.operations.rpc import RPCError
from ncclient.xml_ import to_ele, replace_namespace, BASE_NS_1_0, NETCONF_MONITORING_NS
from ncclient.transport.third_party.junos.parser import JunosXMLParser
from ncclient.transport.parser import DefaultXMLParser
from ncclient.transport.parser import SAXParserHandler
class JunosDeviceHandler(DefaultDeviceHandler):
    """
    Juniper handler for device specific information.

    """

    def __init__(self, device_params):
        super(JunosDeviceHandler, self).__init__(device_params)
        self.__reply_parsing_error_transform_by_cls = {GetSchemaReply: fix_get_schema_reply}

    def add_additional_operations(self):
        dict = {}
        dict['rpc'] = ExecuteRpc
        dict['get_configuration'] = GetConfiguration
        dict['load_configuration'] = LoadConfiguration
        dict['compare_configuration'] = CompareConfiguration
        dict['command'] = Command
        dict['reboot'] = Reboot
        dict['halt'] = Halt
        dict['commit'] = Commit
        dict['rollback'] = Rollback
        return dict

    def perform_qualify_check(self):
        return False

    def handle_raw_dispatch(self, raw):
        if 'routing-engine' in raw:
            raw = re.sub('<ok/>', '</routing-engine>\n<ok/>', raw)
            return raw
        elif re.search('<rpc-reply>.*?</rpc-reply>.*</hello>?', raw, re.M | re.S):
            errs = re.findall('<rpc-error>.*?</rpc-error>', raw, re.M | re.S)
            err_list = []
            if errs:
                add_ns = '\n                        <xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">\n                          <xsl:output indent="yes"/>\n                            <xsl:template match="*">\n                            <xsl:element name="{local-name()}" namespace="urn:ietf:params:xml:ns:netconf:base:1.0">\n                            <xsl:apply-templates select="@*|node()"/>\n                            </xsl:element>\n                          </xsl:template>\n                        </xsl:stylesheet>'
                for err in errs:
                    doc = etree.ElementTree(etree.XML(err))
                    xslt = etree.XSLT(etree.XML(add_ns))
                    transformed_xml = etree.XML(etree.tostring(xslt(doc)))
                    err_list.append(RPCError(transformed_xml))
                return RPCError(to_ele('<rpc-reply>' + ''.join(errs) + '</rpc-reply>'), err_list)
        else:
            return False

    def handle_connection_exceptions(self, sshsession):
        c = sshsession._channel = sshsession._transport.open_channel(kind='session')
        c.set_name('netconf-command-' + str(sshsession._channel_id))
        c.exec_command('xml-mode netconf need-trailer')
        return True

    def reply_parsing_error_transform(self, reply_cls):
        return self.__reply_parsing_error_transform_by_cls.get(reply_cls)

    def transform_reply(self):
        reply = '<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">\n        <xsl:output method="xml" indent="no"/>\n\n        <xsl:template match="/|comment()|processing-instruction()">\n            <xsl:copy>\n                <xsl:apply-templates/>\n            </xsl:copy>\n        </xsl:template>\n\n        <xsl:template match="*">\n            <xsl:element name="{local-name()}">\n                <xsl:apply-templates select="@*|node()"/>\n            </xsl:element>\n        </xsl:template>\n\n        <xsl:template match="@*">\n            <xsl:attribute name="{local-name()}">\n                <xsl:value-of select="."/>\n            </xsl:attribute>\n        </xsl:template>\n        </xsl:stylesheet>\n        '
        import sys
        if sys.version < '3':
            return reply
        else:
            return reply.encode('UTF-8')

    def get_xml_parser(self, session):
        if self.device_params.get('use_filter', False):
            l = session.get_listener_instance(SAXParserHandler)
            if l:
                session.remove_listener(l)
                del l
            session.add_listener(SAXParserHandler(session))
            return JunosXMLParser(session)
        else:
            return DefaultXMLParser(session)