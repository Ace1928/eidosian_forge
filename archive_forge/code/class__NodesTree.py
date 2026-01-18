import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
class _NodesTree:

    def __init__(self, module):
        self._base_node = _NodesTreeNode(module)
        self._working_stack = [self._base_node]
        self._module = module
        self._prefix_remainder = ''
        self.prefix = ''
        self.indents = [0]

    @property
    def parsed_until_line(self):
        return self._working_stack[-1].get_last_line(self.prefix)

    def _update_insertion_node(self, indentation):
        for node in reversed(list(self._working_stack)):
            if node.indentation < indentation or node is self._working_stack[0]:
                return node
            self._working_stack.pop()

    def add_parsed_nodes(self, tree_nodes, keyword_token_indents):
        old_prefix = self.prefix
        tree_nodes = self._remove_endmarker(tree_nodes)
        if not tree_nodes:
            self.prefix = old_prefix + self.prefix
            return
        assert tree_nodes[0].type != 'newline'
        node = self._update_insertion_node(tree_nodes[0].start_pos[1])
        assert node.tree_node.type in ('suite', 'file_input')
        node.add_tree_nodes(old_prefix, tree_nodes)
        self._update_parsed_node_tos(tree_nodes[-1], keyword_token_indents)

    def _update_parsed_node_tos(self, tree_node, keyword_token_indents):
        if tree_node.type == 'suite':
            def_leaf = tree_node.parent.children[0]
            new_tos = _NodesTreeNode(tree_node, indentation=keyword_token_indents[def_leaf.start_pos][-1])
            new_tos.add_tree_nodes('', list(tree_node.children))
            self._working_stack[-1].add_child_node(new_tos)
            self._working_stack.append(new_tos)
            self._update_parsed_node_tos(tree_node.children[-1], keyword_token_indents)
        elif _func_or_class_has_suite(tree_node):
            self._update_parsed_node_tos(tree_node.children[-1], keyword_token_indents)

    def _remove_endmarker(self, tree_nodes):
        """
        Helps cleaning up the tree nodes that get inserted.
        """
        last_leaf = tree_nodes[-1].get_last_leaf()
        is_endmarker = last_leaf.type == 'endmarker'
        self._prefix_remainder = ''
        if is_endmarker:
            prefix = last_leaf.prefix
            separation = max(prefix.rfind('\n'), prefix.rfind('\r'))
            if separation > -1:
                last_leaf.prefix, self._prefix_remainder = (last_leaf.prefix[:separation + 1], last_leaf.prefix[separation + 1:])
        self.prefix = ''
        if is_endmarker:
            self.prefix = last_leaf.prefix
            tree_nodes = tree_nodes[:-1]
        return tree_nodes

    def _get_matching_indent_nodes(self, tree_nodes, is_new_suite):
        node_iterator = iter(tree_nodes)
        if is_new_suite:
            yield next(node_iterator)
        first_node = next(node_iterator)
        indent = _get_indentation(first_node)
        if not is_new_suite and indent not in self.indents:
            return
        yield first_node
        for n in node_iterator:
            if _get_indentation(n) != indent:
                return
            yield n

    def copy_nodes(self, tree_nodes, until_line, line_offset):
        """
        Copies tree nodes from the old parser tree.

        Returns the number of tree nodes that were copied.
        """
        if tree_nodes[0].type in ('error_leaf', 'error_node'):
            return []
        indentation = _get_indentation(tree_nodes[0])
        old_working_stack = list(self._working_stack)
        old_prefix = self.prefix
        old_indents = self.indents
        self.indents = [i for i in self.indents if i <= indentation]
        self._update_insertion_node(indentation)
        new_nodes, self._working_stack, self.prefix, added_indents = self._copy_nodes(list(self._working_stack), tree_nodes, until_line, line_offset, self.prefix)
        if new_nodes:
            self.indents += added_indents
        else:
            self._working_stack = old_working_stack
            self.prefix = old_prefix
            self.indents = old_indents
        return new_nodes

    def _copy_nodes(self, working_stack, nodes, until_line, line_offset, prefix='', is_nested=False):
        new_nodes = []
        added_indents = []
        nodes = list(self._get_matching_indent_nodes(nodes, is_new_suite=is_nested))
        new_prefix = ''
        for node in nodes:
            if node.start_pos[0] > until_line:
                break
            if node.type == 'endmarker':
                break
            if node.type == 'error_leaf' and node.token_type in ('DEDENT', 'ERROR_DEDENT'):
                break
            if _get_last_line(node) > until_line:
                if _func_or_class_has_suite(node):
                    new_nodes.append(node)
                break
            try:
                c = node.children
            except AttributeError:
                pass
            else:
                n = node
                if n.type == 'decorated':
                    n = n.children[-1]
                if n.type in ('async_funcdef', 'async_stmt'):
                    n = n.children[-1]
                if n.type in ('classdef', 'funcdef'):
                    suite_node = n.children[-1]
                else:
                    suite_node = c[-1]
                if suite_node.type in ('error_leaf', 'error_node'):
                    break
            new_nodes.append(node)
        if new_nodes:
            while new_nodes:
                last_node = new_nodes[-1]
                if last_node.type in ('error_leaf', 'error_node') or _is_flow_node(new_nodes[-1]):
                    new_prefix = ''
                    new_nodes.pop()
                    while new_nodes:
                        last_node = new_nodes[-1]
                        if last_node.get_last_leaf().type == 'newline':
                            break
                        new_nodes.pop()
                    continue
                if len(new_nodes) > 1 and new_nodes[-2].type == 'error_node':
                    new_nodes.pop()
                    continue
                break
        if not new_nodes:
            return ([], working_stack, prefix, added_indents)
        tos = working_stack[-1]
        last_node = new_nodes[-1]
        had_valid_suite_last = False
        if _func_or_class_has_suite(last_node):
            suite = last_node
            while suite.type != 'suite':
                suite = suite.children[-1]
            indent = _get_suite_indentation(suite)
            added_indents.append(indent)
            suite_tos = _NodesTreeNode(suite, indentation=_get_indentation(last_node))
            suite_nodes, new_working_stack, new_prefix, ai = self._copy_nodes(working_stack + [suite_tos], suite.children, until_line, line_offset, is_nested=True)
            added_indents += ai
            if len(suite_nodes) < 2:
                new_nodes.pop()
                new_prefix = ''
            else:
                assert new_nodes
                tos.add_child_node(suite_tos)
                working_stack = new_working_stack
                had_valid_suite_last = True
        if new_nodes:
            if not _ends_with_newline(new_nodes[-1].get_last_leaf()) and (not had_valid_suite_last):
                p = new_nodes[-1].get_next_leaf().prefix
                new_prefix = split_lines(p, keepends=True)[0]
            if had_valid_suite_last:
                last = new_nodes[-1]
                if last.type == 'decorated':
                    last = last.children[-1]
                if last.type in ('async_funcdef', 'async_stmt'):
                    last = last.children[-1]
                last_line_offset_leaf = last.children[-2].get_last_leaf()
                assert last_line_offset_leaf == ':'
            else:
                last_line_offset_leaf = new_nodes[-1].get_last_leaf()
            tos.add_tree_nodes(prefix, new_nodes, line_offset, last_line_offset_leaf)
            prefix = new_prefix
            self._prefix_remainder = ''
        return (new_nodes, working_stack, prefix, added_indents)

    def close(self):
        self._base_node.finish()
        try:
            last_leaf = self._module.get_last_leaf()
        except IndexError:
            end_pos = [1, 0]
        else:
            last_leaf = _skip_dedent_error_leaves(last_leaf)
            end_pos = list(last_leaf.end_pos)
        lines = split_lines(self.prefix)
        assert len(lines) > 0
        if len(lines) == 1:
            if lines[0].startswith(BOM_UTF8_STRING) and end_pos == [1, 0]:
                end_pos[1] -= 1
            end_pos[1] += len(lines[0])
        else:
            end_pos[0] += len(lines) - 1
            end_pos[1] = len(lines[-1])
        endmarker = EndMarker('', tuple(end_pos), self.prefix + self._prefix_remainder)
        endmarker.parent = self._module
        self._module.children.append(endmarker)