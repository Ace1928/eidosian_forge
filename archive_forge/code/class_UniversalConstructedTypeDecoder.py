from pyasn1 import debug
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat.integer import from_bytes
from pyasn1.compat.octets import oct2int, octs2ints, ints2octs, null
from pyasn1.type import base
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import tagmap
from pyasn1.type import univ
from pyasn1.type import useful
class UniversalConstructedTypeDecoder(AbstractConstructedDecoder):
    protoRecordComponent = None
    protoSequenceComponent = None

    def _getComponentTagMap(self, asn1Object, idx):
        raise NotImplementedError()

    def _getComponentPositionByType(self, asn1Object, tagSet, idx):
        raise NotImplementedError()

    def _decodeComponents(self, substrate, tagSet=None, decodeFun=None, **options):
        components = []
        componentTypes = set()
        while substrate:
            component, substrate = decodeFun(substrate, **options)
            if component is eoo.endOfOctets:
                break
            components.append(component)
            componentTypes.add(component.tagSet)
        if len(componentTypes) > 1:
            protoComponent = self.protoRecordComponent
        else:
            protoComponent = self.protoSequenceComponent
        asn1Object = protoComponent.clone(tagSet=tag.TagSet(protoComponent.tagSet.baseTag, *tagSet.superTags))
        for idx, component in enumerate(components):
            asn1Object.setComponentByPosition(idx, component, verifyConstraints=False, matchTags=False, matchConstraints=False)
        return (asn1Object, substrate)

    def valueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        if tagSet[0].tagFormat != tag.tagFormatConstructed:
            raise error.PyAsn1Error('Constructed tag format expected')
        head, tail = (substrate[:length], substrate[length:])
        if substrateFun is not None:
            if asn1Spec is not None:
                asn1Object = asn1Spec.clone()
            elif self.protoComponent is not None:
                asn1Object = self.protoComponent.clone(tagSet=tagSet)
            else:
                asn1Object = (self.protoRecordComponent, self.protoSequenceComponent)
            return substrateFun(asn1Object, substrate, length)
        if asn1Spec is None:
            asn1Object, trailing = self._decodeComponents(head, tagSet=tagSet, decodeFun=decodeFun, **options)
            if trailing:
                raise error.PyAsn1Error('Unused trailing %d octets encountered' % len(trailing))
            return (asn1Object, tail)
        asn1Object = asn1Spec.clone()
        if asn1Spec.typeId in (univ.Sequence.typeId, univ.Set.typeId):
            namedTypes = asn1Spec.componentType
            isSetType = asn1Spec.typeId == univ.Set.typeId
            isDeterministic = not isSetType and (not namedTypes.hasOptionalOrDefault)
            seenIndices = set()
            idx = 0
            while head:
                if not namedTypes:
                    componentType = None
                elif isSetType:
                    componentType = namedTypes.tagMapUnique
                else:
                    try:
                        if isDeterministic:
                            componentType = namedTypes[idx].asn1Object
                        elif namedTypes[idx].isOptional or namedTypes[idx].isDefaulted:
                            componentType = namedTypes.getTagMapNearPosition(idx)
                        else:
                            componentType = namedTypes[idx].asn1Object
                    except IndexError:
                        raise error.PyAsn1Error('Excessive components decoded at %r' % (asn1Spec,))
                component, head = decodeFun(head, componentType, **options)
                if not isDeterministic and namedTypes:
                    if isSetType:
                        idx = namedTypes.getPositionByType(component.effectiveTagSet)
                    elif namedTypes[idx].isOptional or namedTypes[idx].isDefaulted:
                        idx = namedTypes.getPositionNearType(component.effectiveTagSet, idx)
                asn1Object.setComponentByPosition(idx, component, verifyConstraints=False, matchTags=False, matchConstraints=False)
                seenIndices.add(idx)
                idx += 1
            if namedTypes:
                if not namedTypes.requiredComponents.issubset(seenIndices):
                    raise error.PyAsn1Error('ASN.1 object %s has uninitialized components' % asn1Object.__class__.__name__)
                if namedTypes.hasOpenTypes:
                    openTypes = options.get('openTypes', {})
                    if openTypes or options.get('decodeOpenTypes', False):
                        for idx, namedType in enumerate(namedTypes.namedTypes):
                            if not namedType.openType:
                                continue
                            if namedType.isOptional and (not asn1Object.getComponentByPosition(idx).isValue):
                                continue
                            governingValue = asn1Object.getComponentByName(namedType.openType.name)
                            try:
                                openType = openTypes[governingValue]
                            except KeyError:
                                try:
                                    openType = namedType.openType[governingValue]
                                except KeyError:
                                    continue
                            component, rest = decodeFun(asn1Object.getComponentByPosition(idx).asOctets(), asn1Spec=openType)
                            asn1Object.setComponentByPosition(idx, component)
            else:
                asn1Object.verifySizeSpec()
        else:
            asn1Object = asn1Spec.clone()
            componentType = asn1Spec.componentType
            idx = 0
            while head:
                component, head = decodeFun(head, componentType, **options)
                asn1Object.setComponentByPosition(idx, component, verifyConstraints=False, matchTags=False, matchConstraints=False)
                idx += 1
        return (asn1Object, tail)

    def indefLenValueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        if tagSet[0].tagFormat != tag.tagFormatConstructed:
            raise error.PyAsn1Error('Constructed tag format expected')
        if substrateFun is not None:
            if asn1Spec is not None:
                asn1Object = asn1Spec.clone()
            elif self.protoComponent is not None:
                asn1Object = self.protoComponent.clone(tagSet=tagSet)
            else:
                asn1Object = (self.protoRecordComponent, self.protoSequenceComponent)
            return substrateFun(asn1Object, substrate, length)
        if asn1Spec is None:
            return self._decodeComponents(substrate, tagSet=tagSet, decodeFun=decodeFun, allowEoo=True, **options)
        asn1Object = asn1Spec.clone()
        if asn1Spec.typeId in (univ.Sequence.typeId, univ.Set.typeId):
            namedTypes = asn1Object.componentType
            isSetType = asn1Object.typeId == univ.Set.typeId
            isDeterministic = not isSetType and (not namedTypes.hasOptionalOrDefault)
            seenIndices = set()
            idx = 0
            while substrate:
                if len(namedTypes) <= idx:
                    asn1Spec = None
                elif isSetType:
                    asn1Spec = namedTypes.tagMapUnique
                else:
                    try:
                        if isDeterministic:
                            asn1Spec = namedTypes[idx].asn1Object
                        elif namedTypes[idx].isOptional or namedTypes[idx].isDefaulted:
                            asn1Spec = namedTypes.getTagMapNearPosition(idx)
                        else:
                            asn1Spec = namedTypes[idx].asn1Object
                    except IndexError:
                        raise error.PyAsn1Error('Excessive components decoded at %r' % (asn1Object,))
                component, substrate = decodeFun(substrate, asn1Spec, allowEoo=True, **options)
                if component is eoo.endOfOctets:
                    break
                if not isDeterministic and namedTypes:
                    if isSetType:
                        idx = namedTypes.getPositionByType(component.effectiveTagSet)
                    elif namedTypes[idx].isOptional or namedTypes[idx].isDefaulted:
                        idx = namedTypes.getPositionNearType(component.effectiveTagSet, idx)
                asn1Object.setComponentByPosition(idx, component, verifyConstraints=False, matchTags=False, matchConstraints=False)
                seenIndices.add(idx)
                idx += 1
            else:
                raise error.SubstrateUnderrunError('No EOO seen before substrate ends')
            if namedTypes:
                if not namedTypes.requiredComponents.issubset(seenIndices):
                    raise error.PyAsn1Error('ASN.1 object %s has uninitialized components' % asn1Object.__class__.__name__)
                if namedTypes.hasOpenTypes:
                    openTypes = options.get('openTypes', None)
                    if openTypes or options.get('decodeOpenTypes', False):
                        for idx, namedType in enumerate(namedTypes.namedTypes):
                            if not namedType.openType:
                                continue
                            if namedType.isOptional and (not asn1Object.getComponentByPosition(idx).isValue):
                                continue
                            governingValue = asn1Object.getComponentByName(namedType.openType.name)
                            try:
                                openType = openTypes[governingValue]
                            except KeyError:
                                try:
                                    openType = namedType.openType[governingValue]
                                except KeyError:
                                    continue
                            component, rest = decodeFun(asn1Object.getComponentByPosition(idx).asOctets(), asn1Spec=openType, allowEoo=True)
                            if component is not eoo.endOfOctets:
                                asn1Object.setComponentByPosition(idx, component)
                else:
                    asn1Object.verifySizeSpec()
        else:
            asn1Object = asn1Spec.clone()
            componentType = asn1Spec.componentType
            idx = 0
            while substrate:
                component, substrate = decodeFun(substrate, componentType, allowEoo=True, **options)
                if component is eoo.endOfOctets:
                    break
                asn1Object.setComponentByPosition(idx, component, verifyConstraints=False, matchTags=False, matchConstraints=False)
                idx += 1
            else:
                raise error.SubstrateUnderrunError('No EOO seen before substrate ends')
        return (asn1Object, substrate)