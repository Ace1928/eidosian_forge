from glanceclient.v1.apiclient import base
def _list_by_member(self, member):
    member_id = base.getid(member)
    url = '/v1/shared-images/%s' % member_id
    resp, body = self.client.get(url)
    out = []
    for member in body['shared_images']:
        member['member_id'] = member_id
        out.append(ImageMember(self, member, loaded=True))
    return out