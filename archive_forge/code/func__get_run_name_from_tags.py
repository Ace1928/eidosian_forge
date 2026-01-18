def _get_run_name_from_tags(tags):
    for tag in tags:
        if tag.key == MLFLOW_RUN_NAME:
            return tag.value