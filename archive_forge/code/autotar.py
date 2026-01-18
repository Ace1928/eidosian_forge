import tarfile

# Files to include in the archive
files_to_archive = ["file1.txt", "file2.txt", "file3.txt"]

# Name of the tar archive
archive_name = "example.tar"

# Create the tar archive
with tarfile.open(archive_name, "w") as tar:
    for file in files_to_archive:
        tar.add(file)

print("Tar archive created successfully.")

# Extract files from the tar archive
with tarfile.open(archive_name, "r") as tar:
    tar.extractall()

print("Files extracted successfully.")
