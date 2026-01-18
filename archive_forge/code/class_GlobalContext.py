from __future__ import unicode_literals
import collections
import logging
from cmakelang.lint import lintdb
class GlobalContext(object):
    category_names = {'C': 'Convention', 'E': 'Error', 'R': 'Refactor', 'W': 'Warning'}

    def __init__(self, outfile):
        self.outfile = outfile
        self.lintdb = lintdb.get_database()
        self.file_ctxs = {}

    def get_file_ctx(self, infile_path, config):
        if infile_path not in self.file_ctxs:
            self.file_ctxs[infile_path] = FileContext(self, infile_path)
        ctx = self.file_ctxs[infile_path]
        ctx.config = config
        return ctx

    def get_category_counts(self):
        lint_counts = {}
        for _, file_ctx in sorted(self.file_ctxs.items()):
            for record in file_ctx.get_lint():
                category_char = record.spec.idstr[0]
                if category_char not in lint_counts:
                    lint_counts[category_char] = 0
                lint_counts[category_char] += 1
        return lint_counts

    def write_summary(self, outfile):
        outfile.write('Summary\n=======\n')
        outfile.write('files scanned: {:d}\n'.format(len(self.file_ctxs)))
        outfile.write('found lint:\n')
        lint_counts = self.get_category_counts()
        fieldwidth = max((len(name) for _, name in self.category_names.items()))
        fmtstr = '  {:>' + str(fieldwidth) + 's}: {:d}\n'
        for category_char, count in sorted(lint_counts.items()):
            category_name = self.category_names[category_char]
            outfile.write(fmtstr.format(category_name, count))
        outfile.write('\n')