from __future__ import unicode_literals
import collections
import logging
from cmakelang.lint import lintdb
class FileContext(object):

    def __init__(self, global_ctx, infile_path):
        self.global_ctx = global_ctx
        self.infile_path = infile_path
        self.config = None
        self._lint = []
        self._suppressions = set()
        self._supressed_count = collections.defaultdict(int)
        self._suppression_events = []

    def is_idstr(self, idstr):
        return idstr in self.global_ctx.lintdb

    def suppress(self, lineno, idlist):
        """
    Given a list of lint ids, enable a suppression for each one which is not
    already supressed. Return the list of new suppressions
    """
        new_suppressions = []
        for idstr in idlist:
            if idstr in self._suppressions:
                continue
            self._suppressions.add(idstr)
            new_suppressions.append(idstr)
        self._suppression_events.append(SuppressionEvent(lineno, 'add', list(new_suppressions)))
        return new_suppressions

    def unsuppress(self, lineno, idlist):
        for idstr in idlist:
            if idstr not in self._suppressions:
                logger.warning('Unsupressing %s which is not currently surpressed', idstr)
            self._suppressions.discard(idstr)
        self._suppression_events.append(SuppressionEvent(lineno, 'remove', list(idlist)))

    def record_lint(self, idstr, *args, **kwargs):
        if idstr in self.config.lint.disabled_codes:
            self._supressed_count[idstr] += 1
            return
        if idstr in self._suppressions:
            self._supressed_count[idstr] += 1
            return
        spec = self.global_ctx.lintdb[idstr]
        location = kwargs.pop('location', ())
        msg = spec.msgfmt.format(*args, **kwargs)
        record = LintRecord(spec, location, msg)
        self._lint.append(record)

    def get_lint(self):
        """Return lint records in sorted order"""
        records = [record for _, _, _, record in sorted(((record.location, record.spec.idstr, idx, record) for idx, record in enumerate(self._lint)))]
        out = []
        events = list(self._suppression_events)
        active_suppressions = set()
        for record in records:
            if record.location:
                while events and record.location[0] >= events[0].lineno:
                    event = events.pop(0)
                    if event.mode == 'add':
                        for idstr in event.suppressions:
                            active_suppressions.add(idstr)
                    elif event.mode == 'remove':
                        for idstr in event.suppressions:
                            active_suppressions.discard(idstr)
                    else:
                        raise ValueError('Illegal suppression event {}'.format(event.mode))
            if record.spec.idstr in active_suppressions:
                continue
            out.append(record)
        return out

    def writeout(self, outfile):
        for record in self.get_lint():
            outfile.write('{:s}:{}\n'.format(self.infile_path, record))

    def has_lint(self):
        return bool(self.get_lint())