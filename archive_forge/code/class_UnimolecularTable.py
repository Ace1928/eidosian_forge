class UnimolecularTable(_RxnTable):
    """Table of unimolecular reactions in a ReactionSystem

    Parameters
    ----------
    rsys : ReactionSystem
    sinks_sources_disjoint : tuple, None or True
        Colors sinks & sources. When ``True`` :meth:`sinks_sources_disjoint` is called.
    html_cell_label : Reaction formatting callback
        The function takes an integer, a Reaction instance and a dict of Substances as
        parameters and return a string.

    Returns
    -------
    string: html representation
    list: reactions not considered
    """
    _rsys_meth = '_unimolecular_reactions'

    def _html(self, printer, **kwargs):
        if 'substances' not in kwargs:
            kwargs['substances'] = self.substances
        ss = printer._get('substances', **kwargs)
        rows = '\n'.join(('<tr><td>%s</td>%s</tr>' % (printer._print(s), self._cell_html(printer, self.idx_rxn_pairs, rowi)) for rowi, s in enumerate(ss.values())))
        return '<table>%s</table>' % rows