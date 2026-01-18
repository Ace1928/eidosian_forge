def DC_transform(html, options, state):
    """
    @param html: a DOM node for the top level html element
    @param options: invocation options
    @type options: L{Options<pyRdfa.options>}
    @param state: top level execution state
    @type state: L{State<pyRdfa.state>}
    """
    from ..host import HostLanguage
    if not options.host_language in [HostLanguage.xhtml, HostLanguage.html5, HostLanguage.xhtml5]:
        return
    head = None
    try:
        head = html.getElementsByTagName('head')[0]
    except:
        return
    dcprefixes = {}
    for link in html.getElementsByTagName('link'):
        if link.hasAttribute('rel'):
            rel = link.getAttribute('rel')
            uri = link.getAttribute('href')
            if uri != None and rel != None and rel.startswith('schema.'):
                try:
                    localname = rel.split('.')[1]
                    head.setAttributeNS('', 'xmlns:' + localname, uri)
                    dcprefixes[localname] = uri
                except:
                    pass
    for link in html.getElementsByTagName('link'):
        if link.hasAttribute('rel'):
            newProp = ''
            for rel in link.getAttribute('rel').strip().split():
                if rel.find('.') != -1:
                    key = rel.split('.', 1)[0]
                    lname = rel.split('.', 1)[1]
                    if key in dcprefixes and lname != '':
                        newProp += ' ' + key + ':' + lname
                    else:
                        newProp += ' ' + rel
                else:
                    newProp += ' ' + rel
            link.setAttribute('rel', newProp.strip())
    for meta in html.getElementsByTagName('meta'):
        if meta.hasAttribute('name'):
            newProp = ''
            for name in meta.getAttribute('name').strip().split():
                if name.find('.') != -1:
                    key = name.split('.', 1)[0]
                    lname = name.split('.', 1)[1]
                    if key in dcprefixes and lname != '':
                        newProp += ' ' + key + ':' + lname
                    else:
                        newProp += ' ' + name
                else:
                    newProp += ' ' + name
            meta.setAttribute('property', newProp.strip())