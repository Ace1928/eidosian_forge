from sqlalchemy import MetaData, select, Table, and_, not_
def _mark_all_public_images_with_public_visibility(engine, images):
    with engine.connect() as conn:
        migrated_rows = conn.execute(images.update().values(visibility='public').where(images.c.is_public))
    return migrated_rows.rowcount