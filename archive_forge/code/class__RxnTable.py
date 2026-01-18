class _RxnTable(object):
    _rsys_meth = None

    def __init__(self, idx_rxn_pairs, substances, colors=None, missing=None, missing_color='eee8aa'):
        self.idx_rxn_pairs = idx_rxn_pairs
        self.substances = substances
        self.colors = colors or {}
        self.missing = missing or []
        self.missing_color = missing_color

    @classmethod
    def from_ReactionSystem(cls, rsys, color_categories=True):
        idx_rxn_pairs, unconsidered_ri = getattr(rsys, cls._rsys_meth)()
        colors = rsys._category_colors() if color_categories else {}
        missing = [not rsys.substance_participation(sk) for sk in rsys.substances]
        return (cls(idx_rxn_pairs, rsys.substances, colors=colors, missing=missing), unconsidered_ri)

    def _repr_html_(self):
        from .web import css
        return css(self, substances=self.substances, colors=self.colors)

    def _cell_label_html(self, printer, ori_idx, rxn):
        """Reaction formatting callback. (reaction index -> string)"""
        pretty = rxn.unicode(self.substances, with_param=True, with_name=False)
        return '<a title="%d: %s">%s</a>' % (ori_idx, pretty, printer._print(rxn.name or rxn.param))

    def _cell_html(self, printer, A, ri, ci=None):
        args = []
        if ci is not None and ri > ci:
            r = '-'
        else:
            if ci is None:
                c = A[ri]
                is_missing = self.missing[ri]
            else:
                c = A[ri][ci]
                is_missing = self.missing[ri] or self.missing[ci]
            if c is None:
                r = ''
            else:
                r = ', '.join((self._cell_label_html(printer, *r) for r in c))
            if is_missing:
                args.append('style="background-color: #%s;"' % self.missing_color)
        return '<td %s>%s</td>' % (' '.join(args), r)