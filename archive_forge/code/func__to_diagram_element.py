import railroad
from pip._vendor import pyparsing
import typing
from typing import (
from jinja2 import Template
from io import StringIO
import inspect
@_apply_diagram_item_enhancements
def _to_diagram_element(element: pyparsing.ParserElement, parent: typing.Optional[EditablePartial], lookup: ConverterState=None, vertical: int=None, index: int=0, name_hint: str=None, show_results_names: bool=False, show_groups: bool=False) -> typing.Optional[EditablePartial]:
    """
    Recursively converts a PyParsing Element to a railroad Element
    :param lookup: The shared converter state that keeps track of useful things
    :param index: The index of this element within the parent
    :param parent: The parent of this element in the output tree
    :param vertical: Controls at what point we make a list of elements vertical. If this is an integer (the default),
    it sets the threshold of the number of items before we go vertical. If True, always go vertical, if False, never
    do so
    :param name_hint: If provided, this will override the generated name
    :param show_results_names: bool flag indicating whether to add annotations for results names
    :returns: The converted version of the input element, but as a Partial that hasn't yet been constructed
    :param show_groups: bool flag indicating whether to show groups using bounding box
    """
    exprs = element.recurse()
    name = name_hint or element.customName or element.__class__.__name__
    el_id = id(element)
    element_results_name = element.resultsName
    if not element.customName:
        if isinstance(element, (pyparsing.Located,)):
            if exprs:
                if not exprs[0].customName:
                    propagated_name = name
                else:
                    propagated_name = None
                return _to_diagram_element(element.expr, parent=parent, lookup=lookup, vertical=vertical, index=index, name_hint=propagated_name, show_results_names=show_results_names, show_groups=show_groups)
    if _worth_extracting(element):
        if el_id in lookup:
            looked_up = lookup[el_id]
            looked_up.mark_for_extraction(el_id, lookup, name=name_hint)
            ret = EditablePartial.from_call(railroad.NonTerminal, text=looked_up.name)
            return ret
        elif el_id in lookup.diagrams:
            ret = EditablePartial.from_call(railroad.NonTerminal, text=lookup.diagrams[el_id].kwargs['name'])
            return ret
    if isinstance(element, pyparsing.And):
        if not exprs:
            return None
        if len(set(((e.name, e.resultsName) for e in exprs))) == 1:
            ret = EditablePartial.from_call(railroad.OneOrMore, item='', repeat=str(len(exprs)))
        elif _should_vertical(vertical, exprs):
            ret = EditablePartial.from_call(railroad.Stack, items=[])
        else:
            ret = EditablePartial.from_call(railroad.Sequence, items=[])
    elif isinstance(element, (pyparsing.Or, pyparsing.MatchFirst)):
        if not exprs:
            return None
        if _should_vertical(vertical, exprs):
            ret = EditablePartial.from_call(railroad.Choice, 0, items=[])
        else:
            ret = EditablePartial.from_call(railroad.HorizontalChoice, items=[])
    elif isinstance(element, pyparsing.Each):
        if not exprs:
            return None
        ret = EditablePartial.from_call(EachItem, items=[])
    elif isinstance(element, pyparsing.NotAny):
        ret = EditablePartial.from_call(AnnotatedItem, label='NOT', item='')
    elif isinstance(element, pyparsing.FollowedBy):
        ret = EditablePartial.from_call(AnnotatedItem, label='LOOKAHEAD', item='')
    elif isinstance(element, pyparsing.PrecededBy):
        ret = EditablePartial.from_call(AnnotatedItem, label='LOOKBEHIND', item='')
    elif isinstance(element, pyparsing.Group):
        if show_groups:
            ret = EditablePartial.from_call(AnnotatedItem, label='', item='')
        else:
            ret = EditablePartial.from_call(railroad.Group, label='', item='')
    elif isinstance(element, pyparsing.TokenConverter):
        label = type(element).__name__.lower()
        if label == 'tokenconverter':
            ret = EditablePartial.from_call(railroad.Sequence, items=[])
        else:
            ret = EditablePartial.from_call(AnnotatedItem, label=label, item='')
    elif isinstance(element, pyparsing.Opt):
        ret = EditablePartial.from_call(railroad.Optional, item='')
    elif isinstance(element, pyparsing.OneOrMore):
        ret = EditablePartial.from_call(railroad.OneOrMore, item='')
    elif isinstance(element, pyparsing.ZeroOrMore):
        ret = EditablePartial.from_call(railroad.ZeroOrMore, item='')
    elif isinstance(element, pyparsing.Group):
        ret = EditablePartial.from_call(railroad.Group, item=None, label=element_results_name)
    elif isinstance(element, pyparsing.Empty) and (not element.customName):
        ret = None
    elif isinstance(element, pyparsing.ParseElementEnhance):
        ret = EditablePartial.from_call(railroad.Sequence, items=[])
    elif len(exprs) > 0 and (not element_results_name):
        ret = EditablePartial.from_call(railroad.Group, item='', label=name)
    elif len(exprs) > 0:
        ret = EditablePartial.from_call(railroad.Sequence, items=[])
    else:
        terminal = EditablePartial.from_call(railroad.Terminal, element.defaultName)
        ret = terminal
    if ret is None:
        return
    lookup[el_id] = ElementState(element=element, converted=ret, parent=parent, parent_index=index, number=lookup.generate_index())
    if element.customName:
        lookup[el_id].mark_for_extraction(el_id, lookup, element.customName)
    i = 0
    for expr in exprs:
        if 'items' in ret.kwargs:
            ret.kwargs['items'].insert(i, None)
        item = _to_diagram_element(expr, parent=ret, lookup=lookup, vertical=vertical, index=i, show_results_names=show_results_names, show_groups=show_groups)
        if item is not None:
            if 'item' in ret.kwargs:
                ret.kwargs['item'] = item
            elif 'items' in ret.kwargs:
                ret.kwargs['items'][i] = item
                i += 1
        elif 'items' in ret.kwargs:
            del ret.kwargs['items'][i]
    if ret and ('items' in ret.kwargs and len(ret.kwargs['items']) == 0 or ('item' in ret.kwargs and ret.kwargs['item'] is None)):
        ret = EditablePartial.from_call(railroad.Terminal, name)
    if el_id in lookup:
        lookup[el_id].complete = True
    if el_id in lookup and lookup[el_id].extract and lookup[el_id].complete:
        lookup.extract_into_diagram(el_id)
        if ret is not None:
            ret = EditablePartial.from_call(railroad.NonTerminal, text=lookup.diagrams[el_id].kwargs['name'])
    return ret