import operator
import os
import re
import subprocess
import tempfile
from functools import reduce
from optparse import OptionParser
from nltk.internals import find_binary
from nltk.sem.drt import (
from nltk.sem.logic import (
class BoxerOutputDrsParser(DrtParser):

    def __init__(self, discourse_id=None):
        """
        This class is used to parse the Prolog DRS output from Boxer into a
        hierarchy of python objects.
        """
        DrtParser.__init__(self)
        self.discourse_id = discourse_id
        self.sentence_id_offset = None
        self.quote_chars = [("'", "'", '\\', False)]

    def parse(self, data, signature=None):
        return DrtParser.parse(self, data, signature)

    def get_all_symbols(self):
        return ['(', ')', ',', '[', ']', ':']

    def handle(self, tok, context):
        return self.handle_drs(tok)

    def attempt_adjuncts(self, expression, context):
        return expression

    def parse_condition(self, indices):
        """
        Parse a DRS condition

        :return: list of ``DrtExpression``
        """
        tok = self.token()
        accum = self.handle_condition(tok, indices)
        if accum is None:
            raise UnexpectedTokenException(tok)
        return accum

    def handle_drs(self, tok):
        if tok == 'drs':
            return self.parse_drs()
        elif tok in ['merge', 'smerge']:
            return self._handle_binary_expression(self._make_merge_expression)(None, [])
        elif tok in ['alfa']:
            return self._handle_alfa(self._make_merge_expression)(None, [])

    def handle_condition(self, tok, indices):
        """
        Handle a DRS condition

        :param indices: list of int
        :return: list of ``DrtExpression``
        """
        if tok == 'not':
            return [self._handle_not()]
        if tok == 'or':
            conds = [self._handle_binary_expression(self._make_or_expression)]
        elif tok == 'imp':
            conds = [self._handle_binary_expression(self._make_imp_expression)]
        elif tok == 'eq':
            conds = [self._handle_eq()]
        elif tok == 'prop':
            conds = [self._handle_prop()]
        elif tok == 'pred':
            conds = [self._handle_pred()]
        elif tok == 'named':
            conds = [self._handle_named()]
        elif tok == 'rel':
            conds = [self._handle_rel()]
        elif tok == 'timex':
            conds = self._handle_timex()
        elif tok == 'card':
            conds = [self._handle_card()]
        elif tok == 'whq':
            conds = [self._handle_whq()]
        elif tok == 'duplex':
            conds = [self._handle_duplex()]
        else:
            conds = []
        return sum(([cond(sent_index, word_indices) for cond in conds] for sent_index, word_indices in self._sent_and_word_indices(indices)), [])

    def _handle_not(self):
        self.assertToken(self.token(), '(')
        drs = self.process_next_expression(None)
        self.assertToken(self.token(), ')')
        return BoxerNot(drs)

    def _handle_pred(self):
        self.assertToken(self.token(), '(')
        variable = self.parse_variable()
        self.assertToken(self.token(), ',')
        name = self.token()
        self.assertToken(self.token(), ',')
        pos = self.token()
        self.assertToken(self.token(), ',')
        sense = int(self.token())
        self.assertToken(self.token(), ')')

        def _handle_pred_f(sent_index, word_indices):
            return BoxerPred(self.discourse_id, sent_index, word_indices, variable, name, pos, sense)
        return _handle_pred_f

    def _handle_duplex(self):
        self.assertToken(self.token(), '(')
        ans_types = []
        self.assertToken(self.token(), 'whq')
        self.assertToken(self.token(), ',')
        d1 = self.process_next_expression(None)
        self.assertToken(self.token(), ',')
        ref = self.parse_variable()
        self.assertToken(self.token(), ',')
        d2 = self.process_next_expression(None)
        self.assertToken(self.token(), ')')
        return lambda sent_index, word_indices: BoxerWhq(self.discourse_id, sent_index, word_indices, ans_types, d1, ref, d2)

    def _handle_named(self):
        self.assertToken(self.token(), '(')
        variable = self.parse_variable()
        self.assertToken(self.token(), ',')
        name = self.token()
        self.assertToken(self.token(), ',')
        type = self.token()
        self.assertToken(self.token(), ',')
        sense = self.token()
        self.assertToken(self.token(), ')')
        return lambda sent_index, word_indices: BoxerNamed(self.discourse_id, sent_index, word_indices, variable, name, type, sense)

    def _handle_rel(self):
        self.assertToken(self.token(), '(')
        var1 = self.parse_variable()
        self.assertToken(self.token(), ',')
        var2 = self.parse_variable()
        self.assertToken(self.token(), ',')
        rel = self.token()
        self.assertToken(self.token(), ',')
        sense = int(self.token())
        self.assertToken(self.token(), ')')
        return lambda sent_index, word_indices: BoxerRel(self.discourse_id, sent_index, word_indices, var1, var2, rel, sense)

    def _handle_timex(self):
        self.assertToken(self.token(), '(')
        arg = self.parse_variable()
        self.assertToken(self.token(), ',')
        new_conds = self._handle_time_expression(arg)
        self.assertToken(self.token(), ')')
        return new_conds

    def _handle_time_expression(self, arg):
        tok = self.token()
        self.assertToken(self.token(), '(')
        if tok == 'date':
            conds = self._handle_date(arg)
        elif tok == 'time':
            conds = self._handle_time(arg)
        else:
            return None
        self.assertToken(self.token(), ')')
        return [lambda sent_index, word_indices: BoxerPred(self.discourse_id, sent_index, word_indices, arg, tok, 'n', 0)] + [lambda sent_index, word_indices: cond for cond in conds]

    def _handle_date(self, arg):
        conds = []
        (sent_index, word_indices), = self._sent_and_word_indices(self._parse_index_list())
        self.assertToken(self.token(), '(')
        pol = self.token()
        self.assertToken(self.token(), ')')
        conds.append(BoxerPred(self.discourse_id, sent_index, word_indices, arg, f'date_pol_{pol}', 'a', 0))
        self.assertToken(self.token(), ',')
        (sent_index, word_indices), = self._sent_and_word_indices(self._parse_index_list())
        year = self.token()
        if year != 'XXXX':
            year = year.replace(':', '_')
            conds.append(BoxerPred(self.discourse_id, sent_index, word_indices, arg, f'date_year_{year}', 'a', 0))
        self.assertToken(self.token(), ',')
        (sent_index, word_indices), = self._sent_and_word_indices(self._parse_index_list())
        month = self.token()
        if month != 'XX':
            conds.append(BoxerPred(self.discourse_id, sent_index, word_indices, arg, f'date_month_{month}', 'a', 0))
        self.assertToken(self.token(), ',')
        (sent_index, word_indices), = self._sent_and_word_indices(self._parse_index_list())
        day = self.token()
        if day != 'XX':
            conds.append(BoxerPred(self.discourse_id, sent_index, word_indices, arg, f'date_day_{day}', 'a', 0))
        return conds

    def _handle_time(self, arg):
        conds = []
        self._parse_index_list()
        hour = self.token()
        if hour != 'XX':
            conds.append(self._make_atom('r_hour_2', arg, hour))
        self.assertToken(self.token(), ',')
        self._parse_index_list()
        min = self.token()
        if min != 'XX':
            conds.append(self._make_atom('r_min_2', arg, min))
        self.assertToken(self.token(), ',')
        self._parse_index_list()
        sec = self.token()
        if sec != 'XX':
            conds.append(self._make_atom('r_sec_2', arg, sec))
        return conds

    def _handle_card(self):
        self.assertToken(self.token(), '(')
        variable = self.parse_variable()
        self.assertToken(self.token(), ',')
        value = self.token()
        self.assertToken(self.token(), ',')
        type = self.token()
        self.assertToken(self.token(), ')')
        return lambda sent_index, word_indices: BoxerCard(self.discourse_id, sent_index, word_indices, variable, value, type)

    def _handle_prop(self):
        self.assertToken(self.token(), '(')
        variable = self.parse_variable()
        self.assertToken(self.token(), ',')
        drs = self.process_next_expression(None)
        self.assertToken(self.token(), ')')
        return lambda sent_index, word_indices: BoxerProp(self.discourse_id, sent_index, word_indices, variable, drs)

    def _parse_index_list(self):
        indices = []
        self.assertToken(self.token(), '[')
        while self.token(0) != ']':
            indices.append(self.parse_index())
            if self.token(0) == ',':
                self.token()
        self.token()
        self.assertToken(self.token(), ':')
        return indices

    def parse_drs(self):
        self.assertToken(self.token(), '(')
        self.assertToken(self.token(), '[')
        refs = set()
        while self.token(0) != ']':
            indices = self._parse_index_list()
            refs.add(self.parse_variable())
            if self.token(0) == ',':
                self.token()
        self.token()
        self.assertToken(self.token(), ',')
        self.assertToken(self.token(), '[')
        conds = []
        while self.token(0) != ']':
            indices = self._parse_index_list()
            conds.extend(self.parse_condition(indices))
            if self.token(0) == ',':
                self.token()
        self.token()
        self.assertToken(self.token(), ')')
        return BoxerDrs(list(refs), conds)

    def _handle_binary_expression(self, make_callback):
        self.assertToken(self.token(), '(')
        drs1 = self.process_next_expression(None)
        self.assertToken(self.token(), ',')
        drs2 = self.process_next_expression(None)
        self.assertToken(self.token(), ')')
        return lambda sent_index, word_indices: make_callback(sent_index, word_indices, drs1, drs2)

    def _handle_alfa(self, make_callback):
        self.assertToken(self.token(), '(')
        type = self.token()
        self.assertToken(self.token(), ',')
        drs1 = self.process_next_expression(None)
        self.assertToken(self.token(), ',')
        drs2 = self.process_next_expression(None)
        self.assertToken(self.token(), ')')
        return lambda sent_index, word_indices: make_callback(sent_index, word_indices, drs1, drs2)

    def _handle_eq(self):
        self.assertToken(self.token(), '(')
        var1 = self.parse_variable()
        self.assertToken(self.token(), ',')
        var2 = self.parse_variable()
        self.assertToken(self.token(), ')')
        return lambda sent_index, word_indices: BoxerEq(self.discourse_id, sent_index, word_indices, var1, var2)

    def _handle_whq(self):
        self.assertToken(self.token(), '(')
        self.assertToken(self.token(), '[')
        ans_types = []
        while self.token(0) != ']':
            cat = self.token()
            self.assertToken(self.token(), ':')
            if cat == 'des':
                ans_types.append(self.token())
            elif cat == 'num':
                ans_types.append('number')
                typ = self.token()
                if typ == 'cou':
                    ans_types.append('count')
                else:
                    ans_types.append(typ)
            else:
                ans_types.append(self.token())
        self.token()
        self.assertToken(self.token(), ',')
        d1 = self.process_next_expression(None)
        self.assertToken(self.token(), ',')
        ref = self.parse_variable()
        self.assertToken(self.token(), ',')
        d2 = self.process_next_expression(None)
        self.assertToken(self.token(), ')')
        return lambda sent_index, word_indices: BoxerWhq(self.discourse_id, sent_index, word_indices, ans_types, d1, ref, d2)

    def _make_merge_expression(self, sent_index, word_indices, drs1, drs2):
        return BoxerDrs(drs1.refs + drs2.refs, drs1.conds + drs2.conds)

    def _make_or_expression(self, sent_index, word_indices, drs1, drs2):
        return BoxerOr(self.discourse_id, sent_index, word_indices, drs1, drs2)

    def _make_imp_expression(self, sent_index, word_indices, drs1, drs2):
        return BoxerDrs(drs1.refs, drs1.conds, drs2)

    def parse_variable(self):
        var = self.token()
        assert re.match('^[exps]\\d+$', var), var
        return var

    def parse_index(self):
        return int(self.token())

    def _sent_and_word_indices(self, indices):
        """
        :return: list of (sent_index, word_indices) tuples
        """
        sent_indices = {i / 1000 - 1 for i in indices if i >= 0}
        if sent_indices:
            pairs = []
            for sent_index in sent_indices:
                word_indices = [i % 1000 - 1 for i in indices if sent_index == i / 1000 - 1]
                pairs.append((sent_index, word_indices))
            return pairs
        else:
            word_indices = [i % 1000 - 1 for i in indices]
            return [(None, word_indices)]