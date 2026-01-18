from textwrap import dedent
from typing import Any, Dict, List, Optional, Union
from ..language import DirectiveLocation
class IntrospectionObjectType(WithName):
    kind: Literal['object']
    fields: List[IntrospectionField]
    interfaces: List[SimpleIntrospectionType]