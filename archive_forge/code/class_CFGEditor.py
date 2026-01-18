import re
from tkinter import (
from nltk.draw.tree import TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal, _read_cfg_production, nonterminals
from nltk.tree import Tree
class CFGEditor:
    """
    A dialog window for creating and editing context free grammars.
    ``CFGEditor`` imposes the following restrictions:

    - All nonterminals must be strings consisting of word
      characters.
    - All terminals must be strings consisting of word characters
      and space characters.
    """
    ARROW = SymbolWidget.SYMBOLS['rightarrow']
    _LHS_RE = re.compile('(^\\s*\\w+\\s*)(->|(' + ARROW + '))')
    _ARROW_RE = re.compile('\\s*(->|(' + ARROW + '))\\s*')
    _PRODUCTION_RE = re.compile('(^\\s*\\w+\\s*)' + '(->|(' + ARROW + '))\\s*' + '((\\w+|\'[\\w ]*\'|\\"[\\w ]*\\"|\\|)\\s*)*$')
    _TOKEN_RE = re.compile('\\w+|->|\'[\\w ]+\'|"[\\w ]+"|(' + ARROW + ')')
    _BOLD = ('helvetica', -12, 'bold')

    def __init__(self, parent, cfg=None, set_cfg_callback=None):
        self._parent = parent
        if cfg is not None:
            self._cfg = cfg
        else:
            self._cfg = CFG(Nonterminal('S'), [])
        self._set_cfg_callback = set_cfg_callback
        self._highlight_matching_nonterminals = 1
        self._top = Toplevel(parent)
        self._init_bindings()
        self._init_startframe()
        self._startframe.pack(side='top', fill='x', expand=0)
        self._init_prodframe()
        self._prodframe.pack(side='top', fill='both', expand=1)
        self._init_buttons()
        self._buttonframe.pack(side='bottom', fill='x', expand=0)
        self._textwidget.focus()

    def _init_startframe(self):
        frame = self._startframe = Frame(self._top)
        self._start = Entry(frame)
        self._start.pack(side='right')
        Label(frame, text='Start Symbol:').pack(side='right')
        Label(frame, text='Productions:').pack(side='left')
        self._start.insert(0, self._cfg.start().symbol())

    def _init_buttons(self):
        frame = self._buttonframe = Frame(self._top)
        Button(frame, text='Ok', command=self._ok, underline=0, takefocus=0).pack(side='left')
        Button(frame, text='Apply', command=self._apply, underline=0, takefocus=0).pack(side='left')
        Button(frame, text='Reset', command=self._reset, underline=0, takefocus=0).pack(side='left')
        Button(frame, text='Cancel', command=self._cancel, underline=0, takefocus=0).pack(side='left')
        Button(frame, text='Help', command=self._help, underline=0, takefocus=0).pack(side='right')

    def _init_bindings(self):
        self._top.title('CFG Editor')
        self._top.bind('<Control-q>', self._cancel)
        self._top.bind('<Alt-q>', self._cancel)
        self._top.bind('<Control-d>', self._cancel)
        self._top.bind('<Alt-x>', self._cancel)
        self._top.bind('<Escape>', self._cancel)
        self._top.bind('<Alt-c>', self._cancel)
        self._top.bind('<Control-o>', self._ok)
        self._top.bind('<Alt-o>', self._ok)
        self._top.bind('<Control-a>', self._apply)
        self._top.bind('<Alt-a>', self._apply)
        self._top.bind('<Control-r>', self._reset)
        self._top.bind('<Alt-r>', self._reset)
        self._top.bind('<Control-h>', self._help)
        self._top.bind('<Alt-h>', self._help)
        self._top.bind('<F1>', self._help)

    def _init_prodframe(self):
        self._prodframe = Frame(self._top)
        self._textwidget = Text(self._prodframe, background='#e0e0e0', exportselection=1)
        self._textscroll = Scrollbar(self._prodframe, takefocus=0, orient='vertical')
        self._textwidget.config(yscrollcommand=self._textscroll.set)
        self._textscroll.config(command=self._textwidget.yview)
        self._textscroll.pack(side='right', fill='y')
        self._textwidget.pack(expand=1, fill='both', side='left')
        self._textwidget.tag_config('terminal', foreground='#006000')
        self._textwidget.tag_config('arrow', font='symbol')
        self._textwidget.tag_config('error', background='red')
        self._linenum = 0
        self._top.bind('>', self._replace_arrows)
        self._top.bind('<<Paste>>', self._analyze)
        self._top.bind('<KeyPress>', self._check_analyze)
        self._top.bind('<ButtonPress>', self._check_analyze)

        def cycle(e, textwidget=self._textwidget):
            textwidget.tk_focusNext().focus()
        self._textwidget.bind('<Tab>', cycle)
        prod_tuples = [(p.lhs(), [p.rhs()]) for p in self._cfg.productions()]
        for i in range(len(prod_tuples) - 1, 0, -1):
            if prod_tuples[i][0] == prod_tuples[i - 1][0]:
                if () in prod_tuples[i][1]:
                    continue
                if () in prod_tuples[i - 1][1]:
                    continue
                print(prod_tuples[i - 1][1])
                print(prod_tuples[i][1])
                prod_tuples[i - 1][1].extend(prod_tuples[i][1])
                del prod_tuples[i]
        for lhs, rhss in prod_tuples:
            print(lhs, rhss)
            s = '%s ->' % lhs
            for rhs in rhss:
                for elt in rhs:
                    if isinstance(elt, Nonterminal):
                        s += ' %s' % elt
                    else:
                        s += ' %r' % elt
                s += ' |'
            s = s[:-2] + '\n'
            self._textwidget.insert('end', s)
        self._analyze()

    def _clear_tags(self, linenum):
        """
        Remove all tags (except ``arrow`` and ``sel``) from the given
        line of the text widget used for editing the productions.
        """
        start = '%d.0' % linenum
        end = '%d.end' % linenum
        for tag in self._textwidget.tag_names():
            if tag not in ('arrow', 'sel'):
                self._textwidget.tag_remove(tag, start, end)

    def _check_analyze(self, *e):
        """
        Check if we've moved to a new line.  If we have, then remove
        all colorization from the line we moved to, and re-colorize
        the line that we moved from.
        """
        linenum = int(self._textwidget.index('insert').split('.')[0])
        if linenum != self._linenum:
            self._clear_tags(linenum)
            self._analyze_line(self._linenum)
            self._linenum = linenum

    def _replace_arrows(self, *e):
        """
        Replace any ``'->'`` text strings with arrows (char \\256, in
        symbol font).  This searches the whole buffer, but is fast
        enough to be done anytime they press '>'.
        """
        arrow = '1.0'
        while True:
            arrow = self._textwidget.search('->', arrow, 'end+1char')
            if arrow == '':
                break
            self._textwidget.delete(arrow, arrow + '+2char')
            self._textwidget.insert(arrow, self.ARROW, 'arrow')
            self._textwidget.insert(arrow, '\t')
        arrow = '1.0'
        while True:
            arrow = self._textwidget.search(self.ARROW, arrow + '+1char', 'end+1char')
            if arrow == '':
                break
            self._textwidget.tag_add('arrow', arrow, arrow + '+1char')

    def _analyze_token(self, match, linenum):
        """
        Given a line number and a regexp match for a token on that
        line, colorize the token.  Note that the regexp match gives us
        the token's text, start index (on the line), and end index (on
        the line).
        """
        if match.group()[0] in '\'"':
            tag = 'terminal'
        elif match.group() in ('->', self.ARROW):
            tag = 'arrow'
        else:
            tag = 'nonterminal_' + match.group()
            if tag not in self._textwidget.tag_names():
                self._init_nonterminal_tag(tag)
        start = '%d.%d' % (linenum, match.start())
        end = '%d.%d' % (linenum, match.end())
        self._textwidget.tag_add(tag, start, end)

    def _init_nonterminal_tag(self, tag, foreground='blue'):
        self._textwidget.tag_config(tag, foreground=foreground, font=CFGEditor._BOLD)
        if not self._highlight_matching_nonterminals:
            return

        def enter(e, textwidget=self._textwidget, tag=tag):
            textwidget.tag_config(tag, background='#80ff80')

        def leave(e, textwidget=self._textwidget, tag=tag):
            textwidget.tag_config(tag, background='')
        self._textwidget.tag_bind(tag, '<Enter>', enter)
        self._textwidget.tag_bind(tag, '<Leave>', leave)

    def _analyze_line(self, linenum):
        """
        Colorize a given line.
        """
        self._clear_tags(linenum)
        line = self._textwidget.get(repr(linenum) + '.0', repr(linenum) + '.end')
        if CFGEditor._PRODUCTION_RE.match(line):

            def analyze_token(match, self=self, linenum=linenum):
                self._analyze_token(match, linenum)
                return ''
            CFGEditor._TOKEN_RE.sub(analyze_token, line)
        elif line.strip() != '':
            self._mark_error(linenum, line)

    def _mark_error(self, linenum, line):
        """
        Mark the location of an error in a line.
        """
        arrowmatch = CFGEditor._ARROW_RE.search(line)
        if not arrowmatch:
            start = '%d.0' % linenum
            end = '%d.end' % linenum
        elif not CFGEditor._LHS_RE.match(line):
            start = '%d.0' % linenum
            end = '%d.%d' % (linenum, arrowmatch.start())
        else:
            start = '%d.%d' % (linenum, arrowmatch.end())
            end = '%d.end' % linenum
        if self._textwidget.compare(start, '==', end):
            start = '%d.0' % linenum
            end = '%d.end' % linenum
        self._textwidget.tag_add('error', start, end)

    def _analyze(self, *e):
        """
        Replace ``->`` with arrows, and colorize the entire buffer.
        """
        self._replace_arrows()
        numlines = int(self._textwidget.index('end').split('.')[0])
        for linenum in range(1, numlines + 1):
            self._analyze_line(linenum)

    def _parse_productions(self):
        """
        Parse the current contents of the textwidget buffer, to create
        a list of productions.
        """
        productions = []
        text = self._textwidget.get('1.0', 'end')
        text = re.sub(self.ARROW, '->', text)
        text = re.sub('\t', ' ', text)
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            productions += _read_cfg_production(line)
        return productions

    def _destroy(self, *e):
        if self._top is None:
            return
        self._top.destroy()
        self._top = None

    def _ok(self, *e):
        self._apply()
        self._destroy()

    def _apply(self, *e):
        productions = self._parse_productions()
        start = Nonterminal(self._start.get())
        cfg = CFG(start, productions)
        if self._set_cfg_callback is not None:
            self._set_cfg_callback(cfg)

    def _reset(self, *e):
        self._textwidget.delete('1.0', 'end')
        for production in self._cfg.productions():
            self._textwidget.insert('end', '%s\n' % production)
        self._analyze()
        if self._set_cfg_callback is not None:
            self._set_cfg_callback(self._cfg)

    def _cancel(self, *e):
        try:
            self._reset()
        except:
            pass
        self._destroy()

    def _help(self, *e):
        try:
            ShowText(self._parent, 'Help: Chart Parser Demo', _CFGEditor_HELP.strip(), width=75, font='fixed')
        except:
            ShowText(self._parent, 'Help: Chart Parser Demo', _CFGEditor_HELP.strip(), width=75)