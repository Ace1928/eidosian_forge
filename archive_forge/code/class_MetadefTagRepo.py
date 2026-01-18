class MetadefTagRepo(object):

    def __init__(self, base, tag_proxy_class=None, tag_proxy_kwargs=None):
        self.base = base
        self.tag_proxy_helper = Helper(tag_proxy_class, tag_proxy_kwargs)

    def get(self, namespace, name):
        meta_tag = self.base.get(namespace, name)
        return self.tag_proxy_helper.proxy(meta_tag)

    def add(self, meta_tag):
        self.base.add(self.tag_proxy_helper.unproxy(meta_tag))

    def add_tags(self, meta_tags, can_append=False):
        tags_list = []
        for meta_tag in meta_tags:
            tags_list.append(self.tag_proxy_helper.unproxy(meta_tag))
        self.base.add_tags(tags_list, can_append)

    def list(self, *args, **kwargs):
        tags = self.base.list(*args, **kwargs)
        return [self.tag_proxy_helper.proxy(meta_tag) for meta_tag in tags]

    def remove(self, item):
        base_item = self.tag_proxy_helper.unproxy(item)
        result = self.base.remove(base_item)
        return self.tag_proxy_helper.proxy(result)

    def save(self, item):
        base_item = self.tag_proxy_helper.unproxy(item)
        result = self.base.save(base_item)
        return self.tag_proxy_helper.proxy(result)