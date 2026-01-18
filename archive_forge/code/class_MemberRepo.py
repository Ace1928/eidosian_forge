class MemberRepo(object):

    def __init__(self, image, base, member_proxy_class=None, member_proxy_kwargs=None):
        self.image = image
        self.base = base
        self.member_proxy_helper = Helper(member_proxy_class, member_proxy_kwargs)

    def get(self, member_id):
        member = self.base.get(member_id)
        return self.member_proxy_helper.proxy(member)

    def add(self, member):
        self.base.add(self.member_proxy_helper.unproxy(member))

    def list(self, *args, **kwargs):
        members = self.base.list(*args, **kwargs)
        return [self.member_proxy_helper.proxy(member) for member in members]

    def remove(self, member):
        base_item = self.member_proxy_helper.unproxy(member)
        result = self.base.remove(base_item)
        return self.member_proxy_helper.proxy(result)

    def save(self, member, from_state=None):
        base_item = self.member_proxy_helper.unproxy(member)
        result = self.base.save(base_item, from_state=from_state)
        return self.member_proxy_helper.proxy(result)