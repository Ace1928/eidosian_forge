import codecs
import typing
from typing import IO, Union
from pyparsing import (
from rdflib.plugins.sparql.parser import (
from rdflib.plugins.sparql.parserutils import Comp, CompValue, Param
from rdflib.query import Result, ResultParser
from rdflib.term import BNode
from rdflib.term import Literal as RDFLiteral
from rdflib.term import URIRef
class TSVResultParser(ResultParser):

    def parse(self, source: IO, content_type: typing.Optional[str]=None) -> Result:
        if isinstance(source.read(0), bytes):
            source = codecs.getreader('utf-8')(source)
        r = Result('SELECT')
        header = source.readline()
        r.vars = list(HEADER.parseString(header.strip(), parseAll=True))
        r.bindings = []
        while True:
            line = source.readline()
            if not line:
                break
            line = line.strip('\n')
            if line == '':
                continue
            row = ROW.parseString(line, parseAll=True)
            r.bindings.append(dict(zip(r.vars, (self.convertTerm(x) for x in row))))
        return r

    def convertTerm(self, t: Union[object, RDFLiteral, BNode, CompValue, URIRef]) -> typing.Optional[Union[object, BNode, URIRef, RDFLiteral]]:
        if t is NONE_VALUE:
            return None
        if isinstance(t, CompValue):
            if t.name == 'literal':
                return RDFLiteral(t.string, lang=t.lang, datatype=t.datatype)
            else:
                raise Exception('I dont know how to handle this: %s' % (t,))
        else:
            return t