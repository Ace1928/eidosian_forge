from sqlalchemy import MetaData, select, Table, and_, not_
def _migrate_all(engine):
    meta = MetaData()
    images = Table('images', meta, autoload_with=engine)
    image_members = Table('image_members', meta, autoload_with=engine)
    num_rows = _mark_all_public_images_with_public_visibility(images)
    num_rows += _mark_all_non_public_images_with_private_visibility(images)
    num_rows += _mark_all_private_images_with_members_as_shared_visibility(images, image_members)
    return num_rows