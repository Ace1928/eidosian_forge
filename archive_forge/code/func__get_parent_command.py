import typer
def _get_parent_command(ctx: typer.Context) -> str:
    parent_command = ''
    ctx_parent = ctx.parent
    while ctx_parent:
        if ctx_parent.info_name:
            parent_command = ctx_parent.info_name + ' ' + parent_command
            ctx_parent = ctx_parent.parent
        else:
            return COMMAND
    return parent_command.strip()