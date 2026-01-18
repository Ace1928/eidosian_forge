from sqlalchemy import MetaData, select, Table, and_, not_
def _mark_all_private_images_with_members_as_shared_visibility(engine, images, image_members):
    with engine.connect() as conn:
        migrated_rows = conn.execute(images.update().values(visibility='shared').where(and_(images.c.visibility == 'private', images.c.id.in_(select(image_members.c.image_id).distinct().where(not_(image_members.c.deleted))))))
    return migrated_rows.rowcount