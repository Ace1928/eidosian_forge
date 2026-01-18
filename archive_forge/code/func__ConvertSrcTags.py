def _ConvertSrcTags(secure_tags):
    template = '    src_secure_tags {{\n      name = "{name}"\n    }}\n'
    records = map(lambda x: template.format(name=x.name), secure_tags)
    return ''.join(records)