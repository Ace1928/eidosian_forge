from __future__ import absolute_import  #, unicode_literals
class StringIOTree(object):
    """
    See module docs.
    """

    def __init__(self, stream=None):
        self.prepended_children = []
        if stream is None:
            stream = StringIO()
        self.stream = stream
        self.write = stream.write
        self.markers = []

    def empty(self):
        if self.stream.tell():
            return False
        return all([child.empty() for child in self.prepended_children]) if self.prepended_children else True

    def getvalue(self):
        content = []
        self._collect_in(content)
        return ''.join(content)

    def _collect_in(self, target_list):
        for x in self.prepended_children:
            x._collect_in(target_list)
        stream_content = self.stream.getvalue()
        if stream_content:
            target_list.append(stream_content)

    def copyto(self, target):
        """Potentially cheaper than getvalue as no string concatenation
        needs to happen."""
        for child in self.prepended_children:
            child.copyto(target)
        stream_content = self.stream.getvalue()
        if stream_content:
            target.write(stream_content)

    def commit(self):
        if self.stream.tell():
            self.prepended_children.append(StringIOTree(self.stream))
            self.prepended_children[-1].markers = self.markers
            self.markers = []
            self.stream = StringIO()
            self.write = self.stream.write

    def reset(self):
        self.prepended_children = []
        self.markers = []
        self.stream = StringIO()
        self.write = self.stream.write

    def insert(self, iotree):
        """
        Insert a StringIOTree (and all of its contents) at this location.
        Further writing to self appears after what is inserted.
        """
        self.commit()
        self.prepended_children.append(iotree)

    def insertion_point(self):
        """
        Returns a new StringIOTree, which is left behind at the current position
        (it what is written to the result will appear right before whatever is
        next written to self).

        Calling getvalue() or copyto() on the result will only return the
        contents written to it.
        """
        self.commit()
        other = StringIOTree()
        self.prepended_children.append(other)
        return other

    def allmarkers(self):
        children = self.prepended_children
        return [m for c in children for m in c.allmarkers()] + self.markers
    '\n    # Print the result of allmarkers in a nice human-readable form. Use it only for debugging.\n    # Prints e.g.\n    # /path/to/source.pyx:\n    #     cython line 2 maps to 3299-3343\n    #     cython line 4 maps to 2236-2245  2306  3188-3201\n    # /path/to/othersource.pyx:\n    #     cython line 3 maps to 1234-1270\n    # ...\n    # Note: In the example above, 3343 maps to line 2, 3344 does not.\n    def print_hr_allmarkers(self):\n        from collections import defaultdict\n        markers = self.allmarkers()\n        totmap = defaultdict(lambda: defaultdict(list))\n        for c_lineno, (cython_desc, cython_lineno) in enumerate(markers):\n            if cython_lineno > 0 and cython_desc.filename is not None:\n                totmap[cython_desc.filename][cython_lineno].append(c_lineno + 1)\n        reprstr = ""\n        if totmap == 0:\n            reprstr += "allmarkers is empty\n"\n        try:\n            sorted(totmap.items())\n        except:\n            print(totmap)\n            print(totmap.items())\n        for cython_path, filemap in sorted(totmap.items()):\n            reprstr += cython_path + ":\n"\n            for cython_lineno, c_linenos in sorted(filemap.items()):\n                reprstr += "\tcython line " + str(cython_lineno) + " maps to "\n                i = 0\n                while i < len(c_linenos):\n                    reprstr += str(c_linenos[i])\n                    flag = False\n                    while i+1 < len(c_linenos) and c_linenos[i+1] == c_linenos[i]+1:\n                        i += 1\n                        flag = True\n                    if flag:\n                        reprstr += "-" + str(c_linenos[i]) + " "\n                    i += 1\n                reprstr += "\n"\n\n        import sys\n        sys.stdout.write(reprstr)\n    '