from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.ttLib import getSearchRange
from .otBase import (
from .otTables import (
from itertools import zip_longest
from functools import partial
import re
import struct
from typing import Optional
import logging
class STXHeader(BaseConverter):

    def __init__(self, name, repeat, aux, tableClass, *, description=''):
        BaseConverter.__init__(self, name, repeat, aux, tableClass, description=description)
        assert issubclass(self.tableClass, AATAction)
        self.classLookup = AATLookup('GlyphClasses', None, None, UShort)
        if issubclass(self.tableClass, ContextualMorphAction):
            self.perGlyphLookup = AATLookup('PerGlyphLookup', None, None, GlyphID)
        else:
            self.perGlyphLookup = None

    def read(self, reader, font, tableDict):
        table = AATStateTable()
        pos = reader.pos
        classTableReader = reader.getSubReader(0)
        stateArrayReader = reader.getSubReader(0)
        entryTableReader = reader.getSubReader(0)
        actionReader = None
        ligaturesReader = None
        table.GlyphClassCount = reader.readULong()
        classTableReader.seek(pos + reader.readULong())
        stateArrayReader.seek(pos + reader.readULong())
        entryTableReader.seek(pos + reader.readULong())
        if self.perGlyphLookup is not None:
            perGlyphTableReader = reader.getSubReader(0)
            perGlyphTableReader.seek(pos + reader.readULong())
        if issubclass(self.tableClass, LigatureMorphAction):
            actionReader = reader.getSubReader(0)
            actionReader.seek(pos + reader.readULong())
            ligComponentReader = reader.getSubReader(0)
            ligComponentReader.seek(pos + reader.readULong())
            ligaturesReader = reader.getSubReader(0)
            ligaturesReader.seek(pos + reader.readULong())
            numLigComponents = (ligaturesReader.pos - ligComponentReader.pos) // 2
            assert numLigComponents >= 0
            table.LigComponents = ligComponentReader.readUShortArray(numLigComponents)
            table.Ligatures = self._readLigatures(ligaturesReader, font)
        elif issubclass(self.tableClass, InsertionMorphAction):
            actionReader = reader.getSubReader(0)
            actionReader.seek(pos + reader.readULong())
        table.GlyphClasses = self.classLookup.read(classTableReader, font, tableDict)
        numStates = int((entryTableReader.pos - stateArrayReader.pos) / (table.GlyphClassCount * 2))
        for stateIndex in range(numStates):
            state = AATState()
            table.States.append(state)
            for glyphClass in range(table.GlyphClassCount):
                entryIndex = stateArrayReader.readUShort()
                state.Transitions[glyphClass] = self._readTransition(entryTableReader, entryIndex, font, actionReader)
        if self.perGlyphLookup is not None:
            table.PerGlyphLookups = self._readPerGlyphLookups(table, perGlyphTableReader, font)
        return table

    def _readTransition(self, reader, entryIndex, font, actionReader):
        transition = self.tableClass()
        entryReader = reader.getSubReader(reader.pos + entryIndex * transition.staticSize)
        transition.decompile(entryReader, font, actionReader)
        return transition

    def _readLigatures(self, reader, font):
        limit = len(reader.data)
        numLigatureGlyphs = (limit - reader.pos) // 2
        return font.getGlyphNameMany(reader.readUShortArray(numLigatureGlyphs))

    def _countPerGlyphLookups(self, table):
        numLookups = 0
        for state in table.States:
            for t in state.Transitions.values():
                if isinstance(t, ContextualMorphAction):
                    if t.MarkIndex != 65535:
                        numLookups = max(numLookups, t.MarkIndex + 1)
                    if t.CurrentIndex != 65535:
                        numLookups = max(numLookups, t.CurrentIndex + 1)
        return numLookups

    def _readPerGlyphLookups(self, table, reader, font):
        pos = reader.pos
        lookups = []
        for _ in range(self._countPerGlyphLookups(table)):
            lookupReader = reader.getSubReader(0)
            lookupReader.seek(pos + reader.readULong())
            lookups.append(self.perGlyphLookup.read(lookupReader, font, {}))
        return lookups

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        glyphClassWriter = OTTableWriter()
        self.classLookup.write(glyphClassWriter, font, tableDict, value.GlyphClasses, repeatIndex=None)
        glyphClassData = pad(glyphClassWriter.getAllData(), 2)
        glyphClassCount = max(value.GlyphClasses.values()) + 1
        glyphClassTableOffset = 16
        if self.perGlyphLookup is not None:
            glyphClassTableOffset += 4
        glyphClassTableOffset += self.tableClass.actionHeaderSize
        actionData, actionIndex = self.tableClass.compileActions(font, value.States)
        stateArrayData, entryTableData = self._compileStates(font, value.States, glyphClassCount, actionIndex)
        stateArrayOffset = glyphClassTableOffset + len(glyphClassData)
        entryTableOffset = stateArrayOffset + len(stateArrayData)
        perGlyphOffset = entryTableOffset + len(entryTableData)
        perGlyphData = pad(self._compilePerGlyphLookups(value, font), 4)
        if actionData is not None:
            actionOffset = entryTableOffset + len(entryTableData)
        else:
            actionOffset = None
        ligaturesOffset, ligComponentsOffset = (None, None)
        ligComponentsData = self._compileLigComponents(value, font)
        ligaturesData = self._compileLigatures(value, font)
        if ligComponentsData is not None:
            assert len(perGlyphData) == 0
            ligComponentsOffset = actionOffset + len(actionData)
            ligaturesOffset = ligComponentsOffset + len(ligComponentsData)
        writer.writeULong(glyphClassCount)
        writer.writeULong(glyphClassTableOffset)
        writer.writeULong(stateArrayOffset)
        writer.writeULong(entryTableOffset)
        if self.perGlyphLookup is not None:
            writer.writeULong(perGlyphOffset)
        if actionOffset is not None:
            writer.writeULong(actionOffset)
        if ligComponentsOffset is not None:
            writer.writeULong(ligComponentsOffset)
            writer.writeULong(ligaturesOffset)
        writer.writeData(glyphClassData)
        writer.writeData(stateArrayData)
        writer.writeData(entryTableData)
        writer.writeData(perGlyphData)
        if actionData is not None:
            writer.writeData(actionData)
        if ligComponentsData is not None:
            writer.writeData(ligComponentsData)
        if ligaturesData is not None:
            writer.writeData(ligaturesData)

    def _compileStates(self, font, states, glyphClassCount, actionIndex):
        stateArrayWriter = OTTableWriter()
        entries, entryIDs = ([], {})
        for state in states:
            for glyphClass in range(glyphClassCount):
                transition = state.Transitions[glyphClass]
                entryWriter = OTTableWriter()
                transition.compile(entryWriter, font, actionIndex)
                entryData = entryWriter.getAllData()
                assert len(entryData) == transition.staticSize, '%s has staticSize %d, but actually wrote %d bytes' % (repr(transition), transition.staticSize, len(entryData))
                entryIndex = entryIDs.get(entryData)
                if entryIndex is None:
                    entryIndex = len(entries)
                    entryIDs[entryData] = entryIndex
                    entries.append(entryData)
                stateArrayWriter.writeUShort(entryIndex)
        stateArrayData = pad(stateArrayWriter.getAllData(), 4)
        entryTableData = pad(bytesjoin(entries), 4)
        return (stateArrayData, entryTableData)

    def _compilePerGlyphLookups(self, table, font):
        if self.perGlyphLookup is None:
            return b''
        numLookups = self._countPerGlyphLookups(table)
        assert len(table.PerGlyphLookups) == numLookups, 'len(AATStateTable.PerGlyphLookups) is %d, but the actions inside the table refer to %d' % (len(table.PerGlyphLookups), numLookups)
        writer = OTTableWriter()
        for lookup in table.PerGlyphLookups:
            lookupWriter = writer.getSubWriter()
            self.perGlyphLookup.write(lookupWriter, font, {}, lookup, None)
            writer.writeSubTable(lookupWriter, offsetSize=4)
        return writer.getAllData()

    def _compileLigComponents(self, table, font):
        if not hasattr(table, 'LigComponents'):
            return None
        writer = OTTableWriter()
        for component in table.LigComponents:
            writer.writeUShort(component)
        return writer.getAllData()

    def _compileLigatures(self, table, font):
        if not hasattr(table, 'Ligatures'):
            return None
        writer = OTTableWriter()
        for glyphName in table.Ligatures:
            writer.writeUShort(font.getGlyphID(glyphName))
        return writer.getAllData()

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.begintag(name, attrs)
        xmlWriter.newline()
        xmlWriter.comment('GlyphClassCount=%s' % value.GlyphClassCount)
        xmlWriter.newline()
        for g, klass in sorted(value.GlyphClasses.items()):
            xmlWriter.simpletag('GlyphClass', glyph=g, value=klass)
            xmlWriter.newline()
        for stateIndex, state in enumerate(value.States):
            xmlWriter.begintag('State', index=stateIndex)
            xmlWriter.newline()
            for glyphClass, trans in sorted(state.Transitions.items()):
                trans.toXML(xmlWriter, font=font, attrs={'onGlyphClass': glyphClass}, name='Transition')
            xmlWriter.endtag('State')
            xmlWriter.newline()
        for i, lookup in enumerate(value.PerGlyphLookups):
            xmlWriter.begintag('PerGlyphLookup', index=i)
            xmlWriter.newline()
            for glyph, val in sorted(lookup.items()):
                xmlWriter.simpletag('Lookup', glyph=glyph, value=val)
                xmlWriter.newline()
            xmlWriter.endtag('PerGlyphLookup')
            xmlWriter.newline()
        if hasattr(value, 'LigComponents'):
            xmlWriter.begintag('LigComponents')
            xmlWriter.newline()
            for i, val in enumerate(getattr(value, 'LigComponents')):
                xmlWriter.simpletag('LigComponent', index=i, value=val)
                xmlWriter.newline()
            xmlWriter.endtag('LigComponents')
            xmlWriter.newline()
        self._xmlWriteLigatures(xmlWriter, font, value, name, attrs)
        xmlWriter.endtag(name)
        xmlWriter.newline()

    def _xmlWriteLigatures(self, xmlWriter, font, value, name, attrs):
        if not hasattr(value, 'Ligatures'):
            return
        xmlWriter.begintag('Ligatures')
        xmlWriter.newline()
        for i, g in enumerate(getattr(value, 'Ligatures')):
            xmlWriter.simpletag('Ligature', index=i, glyph=g)
            xmlWriter.newline()
        xmlWriter.endtag('Ligatures')
        xmlWriter.newline()

    def xmlRead(self, attrs, content, font):
        table = AATStateTable()
        for eltName, eltAttrs, eltContent in filter(istuple, content):
            if eltName == 'GlyphClass':
                glyph = eltAttrs['glyph']
                value = eltAttrs['value']
                table.GlyphClasses[glyph] = safeEval(value)
            elif eltName == 'State':
                state = self._xmlReadState(eltAttrs, eltContent, font)
                table.States.append(state)
            elif eltName == 'PerGlyphLookup':
                lookup = self.perGlyphLookup.xmlRead(eltAttrs, eltContent, font)
                table.PerGlyphLookups.append(lookup)
            elif eltName == 'LigComponents':
                table.LigComponents = self._xmlReadLigComponents(eltAttrs, eltContent, font)
            elif eltName == 'Ligatures':
                table.Ligatures = self._xmlReadLigatures(eltAttrs, eltContent, font)
        table.GlyphClassCount = max(table.GlyphClasses.values()) + 1
        return table

    def _xmlReadState(self, attrs, content, font):
        state = AATState()
        for eltName, eltAttrs, eltContent in filter(istuple, content):
            if eltName == 'Transition':
                glyphClass = safeEval(eltAttrs['onGlyphClass'])
                transition = self.tableClass()
                transition.fromXML(eltName, eltAttrs, eltContent, font)
                state.Transitions[glyphClass] = transition
        return state

    def _xmlReadLigComponents(self, attrs, content, font):
        ligComponents = []
        for eltName, eltAttrs, _eltContent in filter(istuple, content):
            if eltName == 'LigComponent':
                ligComponents.append(safeEval(eltAttrs['value']))
        return ligComponents

    def _xmlReadLigatures(self, attrs, content, font):
        ligs = []
        for eltName, eltAttrs, _eltContent in filter(istuple, content):
            if eltName == 'Ligature':
                ligs.append(eltAttrs['glyph'])
        return ligs