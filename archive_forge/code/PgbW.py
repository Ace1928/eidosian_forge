import os


def add_dotenv_loader(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(subdir, filename)
                with open(filepath, "r+") as f:
                    content = f.read()
                    f.seek(0, 0)
                    f.write(
                        "from dotenv import load_dotenv\nload_dotenv()\n\n" + content
                    )


root_dir = "/home/lloyd/Dropbox/evie_env/cursor/RWKV-Runner/backend-python"
add_dotenv_loader(root_dir)
