from lxml import etree
def _wrap_et_loader(loader):

    def load(href, parse, encoding=None, parser=None):
        return loader(href, parse, encoding)
    return load