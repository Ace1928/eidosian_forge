class MetadefObjectRepo(object):

    def __init__(self, base, object_proxy_class=None, object_proxy_kwargs=None):
        self.base = base
        self.object_proxy_helper = Helper(object_proxy_class, object_proxy_kwargs)

    def get(self, namespace, object_name):
        meta_object = self.base.get(namespace, object_name)
        return self.object_proxy_helper.proxy(meta_object)

    def add(self, meta_object):
        self.base.add(self.object_proxy_helper.unproxy(meta_object))

    def list(self, *args, **kwargs):
        objects = self.base.list(*args, **kwargs)
        return [self.object_proxy_helper.proxy(meta_object) for meta_object in objects]

    def remove(self, item):
        base_item = self.object_proxy_helper.unproxy(item)
        result = self.base.remove(base_item)
        return self.object_proxy_helper.proxy(result)

    def save(self, item):
        base_item = self.object_proxy_helper.unproxy(item)
        result = self.base.save(base_item)
        return self.object_proxy_helper.proxy(result)