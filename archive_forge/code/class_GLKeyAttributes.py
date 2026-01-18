from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc3565
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5652
from pyasn1_modules import rfc5751
from pyasn1_modules import rfc5755
class GLKeyAttributes(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.DefaultedNamedType('rekeyControlledByGLO', univ.Boolean().subtype(value=0, implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.DefaultedNamedType('recipientsNotMutuallyAware', univ.Boolean().subtype(value=1, implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))), namedtype.DefaultedNamedType('duration', univ.Integer().subtype(value=0, implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))), namedtype.DefaultedNamedType('generationCounter', univ.Integer().subtype(value=2, implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3))), namedtype.DefaultedNamedType('requestedAlgorithm', requested_algorithm))