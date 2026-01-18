from __future__ import annotations
import collections
import copy
import itertools
import math
import os
import posixpath
from io import BytesIO, StringIO
from textwrap import indent
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union, cast
from fontTools.misc import etree as ET
from fontTools.misc import plistlib
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import tobytes, tostr
class BaseDocWriter(object):
    _whiteSpace = '    '
    axisDescriptorClass = AxisDescriptor
    discreteAxisDescriptorClass = DiscreteAxisDescriptor
    axisLabelDescriptorClass = AxisLabelDescriptor
    axisMappingDescriptorClass = AxisMappingDescriptor
    locationLabelDescriptorClass = LocationLabelDescriptor
    ruleDescriptorClass = RuleDescriptor
    sourceDescriptorClass = SourceDescriptor
    variableFontDescriptorClass = VariableFontDescriptor
    valueAxisSubsetDescriptorClass = ValueAxisSubsetDescriptor
    rangeAxisSubsetDescriptorClass = RangeAxisSubsetDescriptor
    instanceDescriptorClass = InstanceDescriptor

    @classmethod
    def getAxisDecriptor(cls):
        return cls.axisDescriptorClass()

    @classmethod
    def getAxisMappingDescriptor(cls):
        return cls.axisMappingDescriptorClass()

    @classmethod
    def getSourceDescriptor(cls):
        return cls.sourceDescriptorClass()

    @classmethod
    def getInstanceDescriptor(cls):
        return cls.instanceDescriptorClass()

    @classmethod
    def getRuleDescriptor(cls):
        return cls.ruleDescriptorClass()

    def __init__(self, documentPath, documentObject: DesignSpaceDocument):
        self.path = documentPath
        self.documentObject = documentObject
        self.effectiveFormatTuple = self._getEffectiveFormatTuple()
        self.root = ET.Element('designspace')

    def write(self, pretty=True, encoding='UTF-8', xml_declaration=True):
        self.root.attrib['format'] = '.'.join((str(i) for i in self.effectiveFormatTuple))
        if self.documentObject.axes or self.documentObject.axisMappings or self.documentObject.elidedFallbackName is not None:
            axesElement = ET.Element('axes')
            if self.documentObject.elidedFallbackName is not None:
                axesElement.attrib['elidedfallbackname'] = self.documentObject.elidedFallbackName
            self.root.append(axesElement)
        for axisObject in self.documentObject.axes:
            self._addAxis(axisObject)
        if self.documentObject.axisMappings:
            mappingsElement = None
            lastGroup = object()
            for mappingObject in self.documentObject.axisMappings:
                if getattr(mappingObject, 'groupDescription', None) != lastGroup:
                    if mappingsElement is not None:
                        self.root.findall('.axes')[0].append(mappingsElement)
                    lastGroup = getattr(mappingObject, 'groupDescription', None)
                    mappingsElement = ET.Element('mappings')
                    if lastGroup is not None:
                        mappingsElement.attrib['description'] = lastGroup
                self._addAxisMapping(mappingsElement, mappingObject)
            if mappingsElement is not None:
                self.root.findall('.axes')[0].append(mappingsElement)
        if self.documentObject.locationLabels:
            labelsElement = ET.Element('labels')
            for labelObject in self.documentObject.locationLabels:
                self._addLocationLabel(labelsElement, labelObject)
            self.root.append(labelsElement)
        if self.documentObject.rules:
            if getattr(self.documentObject, 'rulesProcessingLast', False):
                attributes = {'processing': 'last'}
            else:
                attributes = {}
            self.root.append(ET.Element('rules', attributes))
        for ruleObject in self.documentObject.rules:
            self._addRule(ruleObject)
        if self.documentObject.sources:
            self.root.append(ET.Element('sources'))
        for sourceObject in self.documentObject.sources:
            self._addSource(sourceObject)
        if self.documentObject.variableFonts:
            variableFontsElement = ET.Element('variable-fonts')
            for variableFont in self.documentObject.variableFonts:
                self._addVariableFont(variableFontsElement, variableFont)
            self.root.append(variableFontsElement)
        if self.documentObject.instances:
            self.root.append(ET.Element('instances'))
        for instanceObject in self.documentObject.instances:
            self._addInstance(instanceObject)
        if self.documentObject.lib:
            self._addLib(self.root, self.documentObject.lib, 2)
        tree = ET.ElementTree(self.root)
        tree.write(self.path, encoding=encoding, method='xml', xml_declaration=xml_declaration, pretty_print=pretty)

    def _getEffectiveFormatTuple(self):
        """Try to use the version specified in the document, or a sufficiently
        recent version to be able to encode what the document contains.
        """
        minVersion = self.documentObject.formatTuple
        if any((hasattr(axis, 'values') or axis.axisOrdering is not None or axis.axisLabels for axis in self.documentObject.axes)) or self.documentObject.locationLabels or any((source.localisedFamilyName for source in self.documentObject.sources)) or self.documentObject.variableFonts or any((instance.locationLabel or instance.userLocation for instance in self.documentObject.instances)):
            if minVersion < (5, 0):
                minVersion = (5, 0)
        if self.documentObject.axisMappings:
            if minVersion < (5, 1):
                minVersion = (5, 1)
        return minVersion

    def _makeLocationElement(self, locationObject, name=None):
        """Convert Location dict to a locationElement."""
        locElement = ET.Element('location')
        if name is not None:
            locElement.attrib['name'] = name
        validatedLocation = self.documentObject.newDefaultLocation()
        for axisName, axisValue in locationObject.items():
            if axisName in validatedLocation:
                validatedLocation[axisName] = axisValue
        for dimensionName, dimensionValue in validatedLocation.items():
            dimElement = ET.Element('dimension')
            dimElement.attrib['name'] = dimensionName
            if type(dimensionValue) == tuple:
                dimElement.attrib['xvalue'] = self.intOrFloat(dimensionValue[0])
                dimElement.attrib['yvalue'] = self.intOrFloat(dimensionValue[1])
            else:
                dimElement.attrib['xvalue'] = self.intOrFloat(dimensionValue)
            locElement.append(dimElement)
        return (locElement, validatedLocation)

    def intOrFloat(self, num):
        if int(num) == num:
            return '%d' % num
        return ('%f' % num).rstrip('0').rstrip('.')

    def _addRule(self, ruleObject):
        ruleElement = ET.Element('rule')
        if ruleObject.name is not None:
            ruleElement.attrib['name'] = ruleObject.name
        for conditions in ruleObject.conditionSets:
            conditionsetElement = ET.Element('conditionset')
            for cond in conditions:
                if cond.get('minimum') is None and cond.get('maximum') is None:
                    continue
                conditionElement = ET.Element('condition')
                conditionElement.attrib['name'] = cond.get('name')
                if cond.get('minimum') is not None:
                    conditionElement.attrib['minimum'] = self.intOrFloat(cond.get('minimum'))
                if cond.get('maximum') is not None:
                    conditionElement.attrib['maximum'] = self.intOrFloat(cond.get('maximum'))
                conditionsetElement.append(conditionElement)
            if len(conditionsetElement):
                ruleElement.append(conditionsetElement)
        for sub in ruleObject.subs:
            subElement = ET.Element('sub')
            subElement.attrib['name'] = sub[0]
            subElement.attrib['with'] = sub[1]
            ruleElement.append(subElement)
        if len(ruleElement):
            self.root.findall('.rules')[0].append(ruleElement)

    def _addAxis(self, axisObject):
        axisElement = ET.Element('axis')
        axisElement.attrib['tag'] = axisObject.tag
        axisElement.attrib['name'] = axisObject.name
        self._addLabelNames(axisElement, axisObject.labelNames)
        if axisObject.map:
            for inputValue, outputValue in axisObject.map:
                mapElement = ET.Element('map')
                mapElement.attrib['input'] = self.intOrFloat(inputValue)
                mapElement.attrib['output'] = self.intOrFloat(outputValue)
                axisElement.append(mapElement)
        if axisObject.axisOrdering or axisObject.axisLabels:
            labelsElement = ET.Element('labels')
            if axisObject.axisOrdering is not None:
                labelsElement.attrib['ordering'] = str(axisObject.axisOrdering)
            for label in axisObject.axisLabels:
                self._addAxisLabel(labelsElement, label)
            axisElement.append(labelsElement)
        if hasattr(axisObject, 'minimum'):
            axisElement.attrib['minimum'] = self.intOrFloat(axisObject.minimum)
            axisElement.attrib['maximum'] = self.intOrFloat(axisObject.maximum)
        elif hasattr(axisObject, 'values'):
            axisElement.attrib['values'] = ' '.join((self.intOrFloat(v) for v in axisObject.values))
        axisElement.attrib['default'] = self.intOrFloat(axisObject.default)
        if axisObject.hidden:
            axisElement.attrib['hidden'] = '1'
        self.root.findall('.axes')[0].append(axisElement)

    def _addAxisMapping(self, mappingsElement, mappingObject):
        mappingElement = ET.Element('mapping')
        if getattr(mappingObject, 'description', None) is not None:
            mappingElement.attrib['description'] = mappingObject.description
        for what in ('inputLocation', 'outputLocation'):
            whatObject = getattr(mappingObject, what, None)
            if whatObject is None:
                continue
            whatElement = ET.Element(what[:-8])
            mappingElement.append(whatElement)
            for name, value in whatObject.items():
                dimensionElement = ET.Element('dimension')
                dimensionElement.attrib['name'] = name
                dimensionElement.attrib['xvalue'] = self.intOrFloat(value)
                whatElement.append(dimensionElement)
        mappingsElement.append(mappingElement)

    def _addAxisLabel(self, axisElement: ET.Element, label: AxisLabelDescriptor) -> None:
        labelElement = ET.Element('label')
        labelElement.attrib['uservalue'] = self.intOrFloat(label.userValue)
        if label.userMinimum is not None:
            labelElement.attrib['userminimum'] = self.intOrFloat(label.userMinimum)
        if label.userMaximum is not None:
            labelElement.attrib['usermaximum'] = self.intOrFloat(label.userMaximum)
        labelElement.attrib['name'] = label.name
        if label.elidable:
            labelElement.attrib['elidable'] = 'true'
        if label.olderSibling:
            labelElement.attrib['oldersibling'] = 'true'
        if label.linkedUserValue is not None:
            labelElement.attrib['linkeduservalue'] = self.intOrFloat(label.linkedUserValue)
        self._addLabelNames(labelElement, label.labelNames)
        axisElement.append(labelElement)

    def _addLabelNames(self, parentElement, labelNames):
        for languageCode, labelName in sorted(labelNames.items()):
            languageElement = ET.Element('labelname')
            languageElement.attrib[XML_LANG] = languageCode
            languageElement.text = labelName
            parentElement.append(languageElement)

    def _addLocationLabel(self, parentElement: ET.Element, label: LocationLabelDescriptor) -> None:
        labelElement = ET.Element('label')
        labelElement.attrib['name'] = label.name
        if label.elidable:
            labelElement.attrib['elidable'] = 'true'
        if label.olderSibling:
            labelElement.attrib['oldersibling'] = 'true'
        self._addLabelNames(labelElement, label.labelNames)
        self._addLocationElement(labelElement, userLocation=label.userLocation)
        parentElement.append(labelElement)

    def _addLocationElement(self, parentElement, *, designLocation: AnisotropicLocationDict=None, userLocation: SimpleLocationDict=None):
        locElement = ET.Element('location')
        for axis in self.documentObject.axes:
            if designLocation is not None and axis.name in designLocation:
                dimElement = ET.Element('dimension')
                dimElement.attrib['name'] = axis.name
                value = designLocation[axis.name]
                if isinstance(value, tuple):
                    dimElement.attrib['xvalue'] = self.intOrFloat(value[0])
                    dimElement.attrib['yvalue'] = self.intOrFloat(value[1])
                else:
                    dimElement.attrib['xvalue'] = self.intOrFloat(value)
                locElement.append(dimElement)
            elif userLocation is not None and axis.name in userLocation:
                dimElement = ET.Element('dimension')
                dimElement.attrib['name'] = axis.name
                value = userLocation[axis.name]
                dimElement.attrib['uservalue'] = self.intOrFloat(value)
                locElement.append(dimElement)
        if len(locElement) > 0:
            parentElement.append(locElement)

    def _addInstance(self, instanceObject):
        instanceElement = ET.Element('instance')
        if instanceObject.name is not None:
            instanceElement.attrib['name'] = instanceObject.name
        if instanceObject.locationLabel is not None:
            instanceElement.attrib['location'] = instanceObject.locationLabel
        if instanceObject.familyName is not None:
            instanceElement.attrib['familyname'] = instanceObject.familyName
        if instanceObject.styleName is not None:
            instanceElement.attrib['stylename'] = instanceObject.styleName
        if instanceObject.localisedStyleName:
            languageCodes = list(instanceObject.localisedStyleName.keys())
            languageCodes.sort()
            for code in languageCodes:
                if code == 'en':
                    continue
                localisedStyleNameElement = ET.Element('stylename')
                localisedStyleNameElement.attrib[XML_LANG] = code
                localisedStyleNameElement.text = instanceObject.getStyleName(code)
                instanceElement.append(localisedStyleNameElement)
        if instanceObject.localisedFamilyName:
            languageCodes = list(instanceObject.localisedFamilyName.keys())
            languageCodes.sort()
            for code in languageCodes:
                if code == 'en':
                    continue
                localisedFamilyNameElement = ET.Element('familyname')
                localisedFamilyNameElement.attrib[XML_LANG] = code
                localisedFamilyNameElement.text = instanceObject.getFamilyName(code)
                instanceElement.append(localisedFamilyNameElement)
        if instanceObject.localisedStyleMapStyleName:
            languageCodes = list(instanceObject.localisedStyleMapStyleName.keys())
            languageCodes.sort()
            for code in languageCodes:
                if code == 'en':
                    continue
                localisedStyleMapStyleNameElement = ET.Element('stylemapstylename')
                localisedStyleMapStyleNameElement.attrib[XML_LANG] = code
                localisedStyleMapStyleNameElement.text = instanceObject.getStyleMapStyleName(code)
                instanceElement.append(localisedStyleMapStyleNameElement)
        if instanceObject.localisedStyleMapFamilyName:
            languageCodes = list(instanceObject.localisedStyleMapFamilyName.keys())
            languageCodes.sort()
            for code in languageCodes:
                if code == 'en':
                    continue
                localisedStyleMapFamilyNameElement = ET.Element('stylemapfamilyname')
                localisedStyleMapFamilyNameElement.attrib[XML_LANG] = code
                localisedStyleMapFamilyNameElement.text = instanceObject.getStyleMapFamilyName(code)
                instanceElement.append(localisedStyleMapFamilyNameElement)
        if self.effectiveFormatTuple >= (5, 0):
            if instanceObject.locationLabel is None:
                self._addLocationElement(instanceElement, designLocation=instanceObject.designLocation, userLocation=instanceObject.userLocation)
        elif instanceObject.location is not None:
            locationElement, instanceObject.location = self._makeLocationElement(instanceObject.location)
            instanceElement.append(locationElement)
        if instanceObject.filename is not None:
            instanceElement.attrib['filename'] = instanceObject.filename
        if instanceObject.postScriptFontName is not None:
            instanceElement.attrib['postscriptfontname'] = instanceObject.postScriptFontName
        if instanceObject.styleMapFamilyName is not None:
            instanceElement.attrib['stylemapfamilyname'] = instanceObject.styleMapFamilyName
        if instanceObject.styleMapStyleName is not None:
            instanceElement.attrib['stylemapstylename'] = instanceObject.styleMapStyleName
        if self.effectiveFormatTuple < (5, 0):
            if instanceObject.glyphs:
                if instanceElement.findall('.glyphs') == []:
                    glyphsElement = ET.Element('glyphs')
                    instanceElement.append(glyphsElement)
                glyphsElement = instanceElement.findall('.glyphs')[0]
                for glyphName, data in sorted(instanceObject.glyphs.items()):
                    glyphElement = self._writeGlyphElement(instanceElement, instanceObject, glyphName, data)
                    glyphsElement.append(glyphElement)
            if instanceObject.kerning:
                kerningElement = ET.Element('kerning')
                instanceElement.append(kerningElement)
            if instanceObject.info:
                infoElement = ET.Element('info')
                instanceElement.append(infoElement)
        self._addLib(instanceElement, instanceObject.lib, 4)
        self.root.findall('.instances')[0].append(instanceElement)

    def _addSource(self, sourceObject):
        sourceElement = ET.Element('source')
        if sourceObject.filename is not None:
            sourceElement.attrib['filename'] = sourceObject.filename
        if sourceObject.name is not None:
            if sourceObject.name.find('temp_master') != 0:
                sourceElement.attrib['name'] = sourceObject.name
        if sourceObject.familyName is not None:
            sourceElement.attrib['familyname'] = sourceObject.familyName
        if sourceObject.styleName is not None:
            sourceElement.attrib['stylename'] = sourceObject.styleName
        if sourceObject.layerName is not None:
            sourceElement.attrib['layer'] = sourceObject.layerName
        if sourceObject.localisedFamilyName:
            languageCodes = list(sourceObject.localisedFamilyName.keys())
            languageCodes.sort()
            for code in languageCodes:
                if code == 'en':
                    continue
                localisedFamilyNameElement = ET.Element('familyname')
                localisedFamilyNameElement.attrib[XML_LANG] = code
                localisedFamilyNameElement.text = sourceObject.getFamilyName(code)
                sourceElement.append(localisedFamilyNameElement)
        if sourceObject.copyLib:
            libElement = ET.Element('lib')
            libElement.attrib['copy'] = '1'
            sourceElement.append(libElement)
        if sourceObject.copyGroups:
            groupsElement = ET.Element('groups')
            groupsElement.attrib['copy'] = '1'
            sourceElement.append(groupsElement)
        if sourceObject.copyFeatures:
            featuresElement = ET.Element('features')
            featuresElement.attrib['copy'] = '1'
            sourceElement.append(featuresElement)
        if sourceObject.copyInfo or sourceObject.muteInfo:
            infoElement = ET.Element('info')
            if sourceObject.copyInfo:
                infoElement.attrib['copy'] = '1'
            if sourceObject.muteInfo:
                infoElement.attrib['mute'] = '1'
            sourceElement.append(infoElement)
        if sourceObject.muteKerning:
            kerningElement = ET.Element('kerning')
            kerningElement.attrib['mute'] = '1'
            sourceElement.append(kerningElement)
        if sourceObject.mutedGlyphNames:
            for name in sourceObject.mutedGlyphNames:
                glyphElement = ET.Element('glyph')
                glyphElement.attrib['name'] = name
                glyphElement.attrib['mute'] = '1'
                sourceElement.append(glyphElement)
        if self.effectiveFormatTuple >= (5, 0):
            self._addLocationElement(sourceElement, designLocation=sourceObject.location)
        else:
            locationElement, sourceObject.location = self._makeLocationElement(sourceObject.location)
            sourceElement.append(locationElement)
        self.root.findall('.sources')[0].append(sourceElement)

    def _addVariableFont(self, parentElement: ET.Element, vf: VariableFontDescriptor) -> None:
        vfElement = ET.Element('variable-font')
        vfElement.attrib['name'] = vf.name
        if vf.filename is not None:
            vfElement.attrib['filename'] = vf.filename
        if vf.axisSubsets:
            subsetsElement = ET.Element('axis-subsets')
            for subset in vf.axisSubsets:
                subsetElement = ET.Element('axis-subset')
                subsetElement.attrib['name'] = subset.name
                if hasattr(subset, 'userMinimum'):
                    subset = cast(RangeAxisSubsetDescriptor, subset)
                    if subset.userMinimum != -math.inf:
                        subsetElement.attrib['userminimum'] = self.intOrFloat(subset.userMinimum)
                    if subset.userMaximum != math.inf:
                        subsetElement.attrib['usermaximum'] = self.intOrFloat(subset.userMaximum)
                    if subset.userDefault is not None:
                        subsetElement.attrib['userdefault'] = self.intOrFloat(subset.userDefault)
                elif hasattr(subset, 'userValue'):
                    subset = cast(ValueAxisSubsetDescriptor, subset)
                    subsetElement.attrib['uservalue'] = self.intOrFloat(subset.userValue)
                subsetsElement.append(subsetElement)
            vfElement.append(subsetsElement)
        self._addLib(vfElement, vf.lib, 4)
        parentElement.append(vfElement)

    def _addLib(self, parentElement: ET.Element, data: Any, indent_level: int) -> None:
        if not data:
            return
        libElement = ET.Element('lib')
        libElement.append(plistlib.totree(data, indent_level=indent_level))
        parentElement.append(libElement)

    def _writeGlyphElement(self, instanceElement, instanceObject, glyphName, data):
        glyphElement = ET.Element('glyph')
        if data.get('mute'):
            glyphElement.attrib['mute'] = '1'
        if data.get('unicodes') is not None:
            glyphElement.attrib['unicode'] = ' '.join([hex(u) for u in data.get('unicodes')])
        if data.get('instanceLocation') is not None:
            locationElement, data['instanceLocation'] = self._makeLocationElement(data.get('instanceLocation'))
            glyphElement.append(locationElement)
        if glyphName is not None:
            glyphElement.attrib['name'] = glyphName
        if data.get('note') is not None:
            noteElement = ET.Element('note')
            noteElement.text = data.get('note')
            glyphElement.append(noteElement)
        if data.get('masters') is not None:
            mastersElement = ET.Element('masters')
            for m in data.get('masters'):
                masterElement = ET.Element('master')
                if m.get('glyphName') is not None:
                    masterElement.attrib['glyphname'] = m.get('glyphName')
                if m.get('font') is not None:
                    masterElement.attrib['source'] = m.get('font')
                if m.get('location') is not None:
                    locationElement, m['location'] = self._makeLocationElement(m.get('location'))
                    masterElement.append(locationElement)
                mastersElement.append(masterElement)
            glyphElement.append(mastersElement)
        return glyphElement