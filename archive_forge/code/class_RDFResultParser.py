from typing import IO, Any, MutableMapping, Optional, Union
from rdflib.graph import Graph
from rdflib.namespace import RDF, Namespace
from rdflib.query import Result, ResultParser
from rdflib.term import Node, Variable
class RDFResultParser(ResultParser):

    def parse(self, source: Union[IO, Graph], **kwargs: Any) -> Result:
        return RDFResult(source, **kwargs)