import docutils.utils.math.tex2unichar as tex2unichar
class mtable(math):

    def xml_start(self):
        return ['\n<%s>' % self.__class__.__name__]