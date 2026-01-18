from lxml import etree
def _lxml_default_loader(href, parse, encoding=None, parser=None):
    if parse == 'xml':
        data = etree.parse(href, parser).getroot()
    else:
        if '://' in href:
            f = urlopen(href)
        else:
            f = open(href, 'rb')
        data = f.read()
        f.close()
        if not encoding:
            encoding = 'utf-8'
        data = data.decode(encoding)
    return data